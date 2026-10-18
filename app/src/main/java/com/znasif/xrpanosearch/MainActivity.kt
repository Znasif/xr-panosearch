package com.znasif.xrpanosearch

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.SoundPool
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import androidx.xr.arcore.ArDevice
import androidx.xr.arcore.TrackingState
import androidx.xr.compose.platform.LocalSession
import androidx.xr.compose.spatial.Subspace
import androidx.xr.runtime.DeviceTrackingMode
import androidx.xr.runtime.Session
import androidx.xr.runtime.math.Vector3
import androidx.xr.scenecore.scene
import com.znasif.xrpanosearch.cam.CropRect
import com.znasif.xrpanosearch.cam.WorldCamera
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.asin

/**
 * PanoSearch (Full Space). A 12-panel panorama ring (Project_Aura geometry, see [Ring]) surrounds
 * you at eye level. In live capture the panel you face becomes a window onto the world camera;
 * hold still for [DWELL_MS] (ticks speed up) and it freezes into that slice of the panorama (chime)
 * while the full frame is embedded on the V69 NPU (full frame + 3x3 sub-regions). Then search by
 * chip, by voice (audio embedding) or by voice via Whisper; matches get cards along their rays.
 * Logcat tag PanoSearch.
 */
class MainActivity : ComponentActivity() {
    companion object {
        const val DWELL_MS = 1500f
        /** Head must leave a slot by this much past its edge before the next slot takes over. */
        const val HYSTERESIS_DEG = 4f
        const val MAX_PITCH_DEG = 35f
    }

    private lateinit var engine: Engine
    private lateinit var cam: WorldCamera
    private lateinit var world: WorldPanels
    private var session: Session? = null

    // Main-thread state.
    private val captures = HashMap<Int, PanelCapture>()
    private val indexing = HashSet<Int>()
    private val slices = HashMap<Int, Bitmap>()
    private var liveCapture = false
    private var activeSlot = -1
    private var progress = 0f
    private var lastFrameNo = -1L
    /** YUV -> RGB for the live panel runs here so the UI thread never converts frames. */
    private val live = java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "pano-live").apply { isDaemon = true } }
    private var converting = false
    private var recording by mutableStateOf(false)
    private var status by mutableStateOf("starting")
    private var npuStatus by mutableStateOf("NPU: connecting")
    private var queryInfo by mutableStateOf("Capture the room, or replay the recorded sweep.")

    private var head: HeadFrame? = null
    private var ringOrigin: HeadFrame? = null

    private lateinit var sounds: SoundPool
    private var tickId = 0
    private var chimeId = 0
    private var lastTickMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine = Engine(this)
        cam = WorldCamera.get(this)
        world = WorldPanels(this)
        sounds = SoundPool.Builder().setMaxStreams(4).setAudioAttributes(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION).build(),
        ).build()
        runCatching {
            tickId = sounds.load(assets.openFd("sounds/panel_tick.wav"), 1)
            chimeId = sounds.load(assets.openFd("sounds/panel_captured.wav"), 1)
        }.onFailure { Log.w(Engine.TAG, "sounds: ${it.message}") }
        val need = arrayOf(
            Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO,
            "android.permission.HEAD_TRACKING", "android.permission.SCENE_UNDERSTANDING_COARSE",
        ).filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (need.isNotEmpty()) requestPermissions(need.toTypedArray(), 1)
        engine.post {
            val s = engine.connect()
            runOnUiThread { npuStatus = "NPU: $s" }
        }
        // Compose only provides the XR session here; every visible panel is a SceneCore entity.
        setContent {
            val s = LocalSession.current
            LaunchedEffect(s) { if (s != null) trackLoop(s) }
            Subspace {}
        }
    }

    override fun onResume() {
        super.onResume()
        if (liveCapture) cam.start()
    }

    override fun onPause() {
        cam.stop()
        super.onPause()
    }

    override fun onDestroy() {
        world.disposeAll()
        sounds.release()
        live.shutdown()
        super.onDestroy()
    }

    // ------------------------------------------------------------------ main loop

    private suspend fun trackLoop(s: Session) {
        session = s
        runCatching { s.scene.requestFullSpaceMode() }
        val cfg = runCatching { s.configure(s.config.copy(deviceTracking = DeviceTrackingMode.LAST_KNOWN)) }
        Log.i(Engine.TAG, "configure deviceTracking: ${cfg.getOrNull() ?: cfg.exceptionOrNull()}")
        val dev = runCatching { ArDevice.getInstance(s) }.onFailure { Log.e(Engine.TAG, "ArDevice", it) }.getOrNull()
        var lastMs = System.currentTimeMillis()
        while (true) {
            val now = System.currentTimeMillis()
            val dt = (now - lastMs).toFloat(); lastMs = now
            val st = dev?.state?.value
            if (st != null && st.trackingState == TrackingState.TRACKING) {
                val sc = s.scene
                val p = sc.perceptionSpace.transformPoseTo(st.devicePose, sc.activitySpace)
                val hf = HeadFrame(p.translation.v3(), p.right.v3().norm(), p.up.v3().norm(), p.forward.v3().norm())
                head = hf
                if (ringOrigin == null) {
                    ringOrigin = hf
                    world.buildRing(s, hf)
                    world.showControl(s, controlView(), hf)
                }
                world.followControl(hf)
                if (liveCapture) liveStep(hf, dt)
            }
            status = buildString {
                val h = head; val o = ringOrigin
                append(if (h == null || o == null) "head pose: waiting" else "facing %.0f°".format(Ring.relYaw(o, h)))
                append("  |  camera: ").append(if (liveCapture) cam.status() else "off")
                append("  |  panorama ${captures.size}/${Ring.SLOTS}")
                if (indexing.isNotEmpty()) append("  |  indexing ${indexing.size}")
            }
            delay(66)
        }
    }

    private fun pitchDeg(h: HeadFrame): Float =
        Math.toDegrees(asin(h.forward.y.coerceIn(-1f, 1f)).toDouble()).toFloat()

    /** One live-capture tick: pick the faced slot, show the live window, run the dwell countdown. */
    private fun liveStep(h: HeadFrame, dt: Float) {
        val o = ringOrigin ?: return
        val rel = Ring.relYaw(o, h)
        val nearest = Math.floorMod(Math.round(rel / Ring.STEP_DEG), Ring.SLOTS)
        val slot = if (activeSlot < 0 || nearest == activeSlot) nearest
        else if (Engine.angleDiff(rel, activeSlot * Ring.STEP_DEG) > Ring.STEP_DEG / 2f + HYSTERESIS_DEG) nearest
        else activeSlot
        if (slot != activeSlot) {
            if (activeSlot >= 0) restoreCell(activeSlot)
            activeSlot = slot
            progress = 0f
        }
        if (captures.containsKey(slot) || slot in indexing) return

        val frame = cam.snapshot()
        if (frame == null) {
            world.setCell(slot, null, "camera\n${cam.status()}", active = true, progress = 0f)
            return
        }
        val crop = Ring.cropFor(o, slot, h, frame.width, frame.height)
        if (crop == null || abs(pitchDeg(h)) > MAX_PITCH_DEG) {
            progress = 0f
            world.setCell(slot, slices[slot], "look level\nat this panel", active = true, progress = 0f)
            return
        }
        if (frame.frameNumber != lastFrameNo && !converting) {
            lastFrameNo = frame.frameNumber
            converting = true
            live.execute {
                val bmp = frame.toBitmap(crop)
                runOnUiThread {
                    converting = false
                    if (slot == activeSlot && !captures.containsKey(slot) && slot !in indexing) slices[slot] = bmp
                }
            }
        }
        if (frame.meanLuma() < 20.0) {
            progress = 0f
            world.setCell(slot, slices[slot], "too dark", active = true, progress = 0f)
            return
        }
        progress += dt / DWELL_MS
        val now = System.currentTimeMillis()
        if (now - lastTickMs > 260 && progress < 1f) {
            lastTickMs = now
            sounds.play(tickId, 0.5f, 0.5f, 1, 0, (1f + progress * 0.6f).coerceIn(0.5f, 2f))
        }
        world.setCell(slot, slices[slot], "", active = true, progress = progress)
        if (progress >= 1f) {
            sounds.play(chimeId, 0.8f, 0.8f, 1, 0, 1f)
            startIndex(slot, h, "live") { frame.toBitmap(CropRect(0, 0, frame.width, frame.height)) }
        }
    }

    private fun restoreCell(slot: Int) {
        val text = when {
            slot in indexing -> "indexing..."
            captures.containsKey(slot) -> ""
            else -> "${(slot * Ring.STEP_DEG).toInt()}°"
        }
        world.setCell(slot, if (captures.containsKey(slot) || slot in indexing) slices[slot] else null, text, false, 0f)
    }

    /** Indexes slot [slot] on the NPU worker; [frame] (e.g. the YUV conversion) also runs there. */
    private fun startIndex(slot: Int, h: HeadFrame, source: String, frame: () -> Bitmap) {
        indexing += slot
        restoreCell(slot)
        engine.post {
            val c = runCatching { engine.index(slot, frame(), h, source) }
            runOnUiThread {
                indexing -= slot
                c.onSuccess { captures[slot] = it.also { cap -> cap.slice = slices[slot] } }
                    .onFailure { queryInfo = "Index failed: ${it.message}" }
                restoreCell(slot)
            }
        }
    }

    private fun resetRing() {
        val s = session ?: return
        val h = head ?: return
        ringOrigin = h
        captures.clear(); slices.clear(); indexing.clear()
        activeSlot = -1; progress = 0f
        world.clearCards()
        world.buildRing(s, h)
    }

    private fun startLive() {
        resetRing()
        liveCapture = true
        queryInfo = "Turn slowly. The panel you face shows the camera; hold still ~1.5 s to capture it. Camera: ${cam.start()}"
    }

    private fun startReplay() {
        liveCapture = false; cam.stop()
        resetRing()
        val o = ringOrigin ?: return
        val frames = engine.replayFrames()
        queryInfo = "Replaying ${frames.size} recorded slots (${frames.keys.sorted().joinToString { "${(it * Ring.STEP_DEG).toInt()}°" }})."
        for ((slot, f) in frames.toSortedMap()) {
            val bmp = engine.loadAsset("sweep_golden/${f.file}")
            val hf = HeadFrame.fromYawPitch(o.pos, o.yawDeg + f.yawDeg, f.pitchDeg)
            Ring.cropFor(o, slot, hf, bmp.width, bmp.height)?.let { c ->
                slices[slot] = Bitmap.createBitmap(bmp, c.x, c.y, c.w, c.h)
            }
            startIndex(slot, hf, f.file) { bmp }
        }
    }

    private fun runQuery(label: String, embed: () -> Engine.Query) {
        if (captures.isEmpty()) { queryInfo = "Capture or replay the room first."; return }
        queryInfo = "Searching \"$label\" ..."
        val snapshot = captures.values.toList()
        engine.post {
            val r = runCatching { embed().let { q -> q to PanoSearch.search(snapshot, q.emb) } }
            runOnUiThread {
                r.onSuccess { (q, hs) ->
                    session?.let { s -> runCatching { world.showHits(s, hs) }.onFailure { Log.e(Engine.TAG, "cards", it) } }
                    queryInfo = "\"${q.label}\": ${hs.size} match(es), best %.3f  (${q.detail})"
                        .format(hs.firstOrNull()?.score ?: 0f)
                    hs.forEachIndexed { i, h ->
                        Log.i(Engine.TAG, "HIT #${i + 1} ${h.capture.source}/${h.region.label} %.3f dir ${h.dir}".format(h.score))
                    }
                }.onFailure { queryInfo = "Query failed: ${it.message}" }
            }
        }
    }

    private fun voiceQuery(useWhisper: Boolean) {
        if (recording) return
        recording = true
        queryInfo = "Listening (up to 5 s) ..."
        engine.post {
            val pcm = MicRecorder.record()
            runOnUiThread {
                recording = false
                if (pcm.size < 16000 / 4) { queryInfo = "Didn't hear anything."; return@runOnUiThread }
                runQuery(if (useWhisper) "voice via Whisper" else "voice") {
                    if (useWhisper) engine.embedTranscript(pcm) else engine.embedAudio(pcm)
                }
            }
        }
    }

    // ------------------------------------------------------------------ control panel

    private fun controlView() = ComposeView(this).apply {
        setViewTreeLifecycleOwner(this@MainActivity)
        setViewTreeSavedStateRegistryOwner(this@MainActivity)
        setViewTreeViewModelStoreOwner(this@MainActivity)
        setContent { MaterialTheme(colorScheme = darkColorScheme()) { ControlPanel() } }
    }

    @OptIn(ExperimentalLayoutApi::class)
    @Composable
    private fun ControlPanel() {
        Column(
            Modifier.fillMaxSize().background(Color(0xEE1B1F24)).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("PanoSearch", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
            Text(npuStatus, fontSize = 14.sp, color = Color(0xFFB0BEC5))
            Text(status, fontSize = 14.sp, color = Color(0xFFB0BEC5))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { startLive() }) { Text("Capture room (live)") }
                OutlinedButton(onClick = { startReplay() }) { Text("Replay sweep") }
                OutlinedButton(onClick = { world.clearCards() }) { Text("Clear") }
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (q in listOf("laptop", "monitor", "phone", "chair", "door", "window", "bottle", "bag")) {
                    AssistChip(onClick = { runQuery(q) { engine.embedText("a $q") } }, label = { Text(q) })
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { voiceQuery(false) }, enabled = !recording) {
                    Text(if (recording) "Listening..." else "Speak (audio)")
                }
                Button(onClick = { voiceQuery(true) }, enabled = !recording) { Text("Speak (Whisper)") }
            }
            Text(queryInfo, fontSize = 14.sp, color = Color(0xFFFFE082))
        }
    }
}

private fun Vector3.v3() = V3(x, y, z)
