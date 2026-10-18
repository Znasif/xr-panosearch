package com.znasif.xrpanosearch

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Off-head 2D gate (XR activities pause when the headset is doffed): indexes the recorded sweep
 * into the ring on the NPU, runs text queries, and logs ranked hits. Tag PanoSearch, ends with
 * GATE_DONE. Extras: --es queries "a|b|c", --ez dumpPrep true (writes the first frame's CHW input
 * to files/prep_<file>.f32 for comparison against PIL on the host).
 *   adb shell am start -n com.znasif.xrpanosearch/.ReplayGateActivity
 */
class ReplayGateActivity : Activity() {
    private lateinit var engine: Engine
    private lateinit var view: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        view = TextView(this).apply { textSize = 14f; setPadding(32, 32, 32, 32) }
        setContentView(view)
        engine = Engine(this)
        val queries = (intent.getStringExtra("queries")
            ?: "a laptop|a door|a window|a chair|a bookshelf|a television|a plant|a lamp")
            .split('|').map { it.trim() }.filter { it.isNotEmpty() }
        val dumpPrep = intent.getBooleanExtra("dumpPrep", false)
        engine.post { runGate(queries, dumpPrep) }
    }

    private fun say(s: String) {
        Log.i(Engine.TAG, s)
        runOnUiThread { view.append(s + "\n") }
    }

    private fun runGate(queries: List<String>, dumpPrep: Boolean) {
        val st = engine.connect()
        say("NPU: $st")
        if (st != "READY") { say("GATE_DONE FAIL"); return }
        val frames = engine.replayFrames()
        say("replay slots: ${frames.keys.sorted()} -> ${frames.values.map { it.file }}")
        val caps = ArrayList<PanelCapture>()
        for ((slot, f) in frames.toSortedMap()) {
            val bmp = engine.loadAsset("sweep_golden/${f.file}")
            if (dumpPrep && caps.isEmpty()) {
                val (px, _) = ImagePrep.toChw(bmp)
                val bb = ByteBuffer.allocate(px.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                bb.asFloatBuffer().put(px)
                File(getExternalFilesDir(null), "prep_${f.file}.f32").writeBytes(bb.array())
                say("dumped prep_${f.file}.f32")
            }
            val head = HeadFrame.fromYawPitch(V3(0f, 0f, 0f), f.yawDeg, f.pitchDeg)
            caps += engine.index(slot, bmp, head, f.file)
            say("indexed slot $slot (${f.file}) vision ${caps.last().visionMs} ms text ${caps.last().textMs} ms")
        }
        for (q in queries) {
            val qe = engine.embedText(q)
            val hits = PanoSearch.search(caps, qe.emb)
            val top = caps.flatMap { c -> c.regions.map { (r, e) -> Triple(c.source, r.label, PanoSearch.dot(qe.emb, e)) } }
                .sortedByDescending { it.third }.take(3)
            say("Q \"$q\" (${qe.ms} ms): " + hits.joinToString { h ->
                "${h.capture.source}/${h.region.label} %.3f yaw %.0f".format(h.score, h.capture.head.yawDeg)
            } + " | top3 " + top.joinToString { "${it.first}/${it.second} %.3f".format(it.third) })
        }
        say("GATE_DONE PASS")
    }
}
