package com.znasif.xrpanosearch

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.xr.embardiment.npu.NpuClient
import org.json.JSONArray
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * NPU work for PanoSearch: ring-slot indexing (vision + 3x3 grid) and query embedding (text,
 * audio, or Whisper transcript -> text). Every call runs on one background thread because
 * NpuClient calls block and must stay off the main thread.
 */
class Engine(context: Context) {
    companion object {
        const val TAG = "PanoSearch"
        const val SLOTS = 12
        const val SLOT_DEG = 360f / SLOTS
        const val QUERY_PREFIX = "task: search result | query: "
        const val THUMB_W = 480

        /** Ring slot whose centre is nearest [yawDeg] (slot 0 = initial forward, +30 deg each, left). */
        fun slotFor(yawDeg: Float): Int = Math.floorMod(Math.round(yawDeg / SLOT_DEG), SLOTS)

        fun slotYaw(slot: Int): Float = slot * SLOT_DEG

        fun angleDiff(a: Float, b: Float): Float {
            var d = (a - b) % 360f
            if (d > 180f) d -= 360f
            if (d < -180f) d += 360f
            return abs(d)
        }
    }

    private val app = context.applicationContext
    val npu = NpuClient(app)
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "pano-npu").apply { isDaemon = true } }

    fun post(block: () -> Unit) = worker.execute {
        try { block() } catch (t: Throwable) { Log.e(TAG, "worker task failed", t) }
    }

    /** Blocking; call on the worker. */
    fun connect(): String = npu.connect(30_000L)

    /** Blocking; call on the worker. Embeds [frame] (any size, 4:3 expected) for ring [slot]. */
    fun index(slot: Int, frame: Bitmap, head: HeadFrame, source: String): PanelCapture {
        val t0 = System.nanoTime()
        val (px, orient) = ImagePrep.toChw(frame)
        val prepMs = (System.nanoTime() - t0) / 1_000_000
        val r = npu.embgemmaImageGrid(px, orient, true)
        val grid = Region.grid3x3(orient == 0)
        val regions = ArrayList<Pair<Region, FloatArray>>(10)
        regions += Region.FULL to r.fullEmbedding
        grid.forEachIndexed { i, g -> regions += g to r.gridEmbeddings[i] }
        val thumbH = THUMB_W * frame.height / frame.width
        val thumb = Bitmap.createScaledBitmap(frame, THUMB_W, thumbH, true)
        Log.i(TAG, "INDEX slot=$slot src=$source yaw=%.1f prep=${prepMs}ms vision=${r.visionUs / 1000}ms text=${r.textUs / 1000}ms wall=${r.wallUs / 1000}ms"
            .format(head.yawDeg))
        return PanelCapture(
            slot, thumb, head, frame.height.toFloat() / frame.width, r.fullEmbedding, regions,
            r.visionUs / 1000, r.textUs / 1000, source,
        )
    }

    class Query(val label: String, val emb: FloatArray, val ms: Long, val detail: String)

    fun embedText(text: String): Query {
        val r = npu.embgemmaText(QUERY_PREFIX + text)
        Log.i(TAG, "QUERY text=\"$text\" tokens=${r.nTokens} npu=${r.textUs / 1000}ms wall=${r.wallUs / 1000}ms")
        return Query(text, r.embedding, r.wallUs / 1000, "text ${r.textUs / 1000} ms")
    }

    /** Spoken query embedded directly from audio (no transcript). */
    fun embedAudio(pcm: ShortArray): Query {
        val bucket = if (pcm.size <= 5 * 16000) 500 else 1120
        val r = npu.embgemmaAudio(pcm, bucket)
        Log.i(TAG, "QUERY audio samples=${pcm.size} soft=${r.nSoftTokens} mel=${r.melUs / 1000}ms audio=${r.audioUs / 1000}ms text=${r.textUs / 1000}ms wall=${r.wallUs / 1000}ms")
        return Query("(voice)", r.embedding, r.wallUs / 1000,
            "audio ${r.audioUs / 1000} ms + text ${r.textUs / 1000} ms")
    }

    /** Spoken query transcribed by Whisper, then embedded as text. */
    fun embedTranscript(pcm: ShortArray): Query {
        val w = npu.whisperTranscribe(pcm)
        val text = w.text.trim().trimEnd('.', '?', '!')
        Log.i(TAG, "WHISPER \"$text\" wall=${w.wallUs / 1000}ms")
        val q = embedText(text)
        return Query(text, q.emb, w.wallUs / 1000 + q.ms, "whisper ${w.wallUs / 1000} ms + ${q.detail}")
    }

    class ReplayFrame(val file: String, val yawDeg: Float, val pitchDeg: Float)

    /**
     * Picks, for each ring slot, the recorded sweep frame closest to the slot centre (within
     * 15 deg). The recorded yaw is relative to that recording's own origin; slot 0 is mapped to
     * the recording's yaw 0.
     */
    fun replayFrames(dir: String = "sweep_golden"): Map<Int, ReplayFrame> {
        val arr = JSONArray(app.assets.open("$dir/poses.json").bufferedReader().readText())
        val best = HashMap<Int, ReplayFrame>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            if (!o.optBoolean("tracking", true)) continue
            val f = ReplayFrame(o.getString("file"), o.getDouble("yawDeg").toFloat(), o.getDouble("pitchDeg").toFloat())
            val s = slotFor(f.yawDeg)
            val d = angleDiff(f.yawDeg, slotYaw(s))
            if (d > 15f) continue
            val cur = best[s]
            if (cur == null || d < angleDiff(cur.yawDeg, slotYaw(s))) best[s] = f
        }
        return best
    }

    fun loadAsset(path: String): Bitmap =
        app.assets.open(path).use { BitmapFactory.decodeStream(it) }
}
