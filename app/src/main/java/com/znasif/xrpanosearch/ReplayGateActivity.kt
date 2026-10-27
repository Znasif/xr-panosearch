package com.znasif.xrpanosearch

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Off-head 2D gate (XR activities pause when the headset is doffed): indexes the recorded sweep
 * into the ring on the NPU, runs text queries, and logs ranked hits. Tag PanoSearch, ends with
 * GATE_DONE. Extras: --es queries "a|b|c", --ez dumpPrep true (writes the first frame's CHW input
 * to files/prep_<file>.f32 for comparison against PIL on the host), --ez golden true (instead
 * compares every embedding against the host fp32 references in files/embgemma2/gate/golden.json,
 * pushed by `tools/setup_device.sh --gate`; passes at cosine >= [GOLDEN_MIN_COS]).
 *   adb shell am start -n com.znasif.xrpanosearch/.ReplayGateActivity [--ez golden true]
 */
class ReplayGateActivity : Activity() {
    companion object {
        const val GOLDEN_MIN_COS = 0.99f
    }

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
        val golden = intent.getBooleanExtra("golden", false)
        engine.post { if (golden) runGolden() else runGate(queries, dumpPrep) }
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
                val px = ImagePrep.toChw(bmp)
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

    private fun runGolden() {
        val st = engine.connect()
        say("NPU: $st")
        if (st != "READY") { say("GATE_DONE FAIL"); return }
        val dir = File(engine.npu.modelDir, "gate")
        val g = JSONObject(File(dir, "golden.json").readText())
        var worst = 1f
        fun check(what: String, got: FloatArray, ref: JSONArray): Float {
            val c = PanoSearch.dot(got, FloatArray(ref.length()) { ref.getDouble(it).toFloat() })
            worst = minOf(worst, c)
            return c
        }
        val frames = g.getJSONArray("frames")
        val visMs = ArrayList<Long>(); val txtMs = ArrayList<Long>(); val wallMs = ArrayList<Long>()
        for (i in 0 until frames.length()) {
            val f = frames.getJSONObject(i)
            val px = ImagePrep.toChw(engine.loadAsset("sweep_golden/${f.getString("file")}"))
            val r = engine.npu.imageGrid(px, true)
            val full = check("full", r.full, f.getJSONArray("full"))
            val grid = f.getJSONArray("grid")
            val gridMin = (0 until 9).minOf { check("grid", r.grid[it], grid.getJSONArray(it)) }
            visMs += r.visionUs / 1000; txtMs += r.textUs / 1000; wallMs += r.wallUs / 1000
            say("IMG ${f.getString("file")} full=%.6f gridMin=%.6f vision=${r.visionUs / 1000}ms text=${r.textUs / 1000}ms wall=${r.wallUs / 1000}ms"
                .format(full, gridMin))
        }
        val queries = g.getJSONArray("queries")
        val qMs = ArrayList<Long>()
        for (i in 0 until queries.length()) {
            val q = queries.getJSONObject(i)
            val r = engine.npu.text(Engine.QUERY_PREFIX + q.getString("text"))
            qMs += r.textUs / 1000
            say("TXT \"${q.getString("text")}\" cos=%.6f tokens=${r.nTokens} npu=${r.textUs / 1000}ms wall=${r.wallUs / 1000}ms"
                .format(check("text", r.embedding, q.getJSONArray("emb"))))
        }
        val audio = g.getJSONArray("audio")
        for (i in 0 until audio.length()) {
            val a = audio.getJSONObject(i)
            val bytes = File(dir, a.getString("pcm")).readBytes()
            val pcm = ShortArray(bytes.size / 2)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pcm)
            val r = engine.npu.audio(pcm)
            say("AUD ${a.getString("pcm")} cos=%.6f soft=${r.nSoftTokens} mel=${r.melUs / 1000}ms audio=${r.audioUs / 1000}ms text=${r.textUs / 1000}ms wall=${r.wallUs / 1000}ms"
                .format(check("audio", r.embedding, a.getJSONArray("emb"))))
        }
        fun med(l: List<Long>) = l.sorted()[l.size / 2]
        say("SUMMARY image median: vision ${med(visMs)} ms, text(1+9) ${med(txtMs)} ms, wall ${med(wallMs)} ms; query text median ${med(qMs)} ms")
        say("GATE_DONE ${if (worst >= GOLDEN_MIN_COS) "PASS" else "FAIL"} minCos=%.6f".format(worst))
    }
}
