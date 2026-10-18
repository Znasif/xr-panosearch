package com.znasif.xrpanosearch

import android.graphics.Bitmap
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** Minimal vector math so capture geometry doesn't depend on quaternion conventions. */
data class V3(val x: Float, val y: Float, val z: Float) {
    operator fun plus(o: V3) = V3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: V3) = V3(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Float) = V3(x * s, y * s, z * s)
    fun dot(o: V3) = x * o.x + y * o.y + z * o.z
    fun cross(o: V3) = V3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
    fun norm(): V3 { val l = sqrt(dot(this)); return if (l > 1e-6f) this * (1f / l) else this }
    override fun toString() = "(%.2f, %.2f, %.2f)".format(x, y, z)
}

/**
 * Head pose at capture time in activity space (OpenXR axes: +X right, +Y up, -Z forward).
 * Stored as an orthonormal frame so any pose source (ArDevice, replayed yaw/pitch) works.
 */
data class HeadFrame(val pos: V3, val right: V3, val up: V3, val forward: V3) {
    companion object {
        /** yawDeg > 0 turns left (counter-clockwise seen from above); pitchDeg > 0 looks up. */
        fun fromYawPitch(pos: V3, yawDeg: Float, pitchDeg: Float): HeadFrame {
            val y = Math.toRadians(yawDeg.toDouble()).toFloat()
            val p = Math.toRadians(pitchDeg.toDouble()).toFloat()
            val fwd = V3(-sin(y) * cos(p), sin(p), -cos(y) * cos(p)).norm()
            val right = V3(cos(y), 0f, -sin(y)).norm()
            val up = right.cross(fwd).norm()
            return HeadFrame(pos, right, up, fwd)
        }
    }

    val yawDeg: Float get() = Math.toDegrees(kotlin.math.atan2(-forward.x, -forward.z).toDouble()).toFloat()

    /**
     * World ray direction through normalized image point (u, v) (0..1, v down) of the world
     * camera. Nominal pinhole: Camera2 camera 0 reports a 1.45 mm lens on a 2.4 mm / 3000 px
     * sensor, i.e. f ~ 0.604 x image width for the 4:3 stream (HFOV ~79 deg). The camera is
     * assumed to look along the head's forward axis; there is no calibrated extrinsic on Android XR.
     */
    fun rayThrough(u: Float, v: Float, aspectHOverW: Float = 0.75f): V3 {
        val xc = (u - 0.5f) / FX_OVER_W
        val yc = -(v - 0.5f) * aspectHOverW / FX_OVER_W
        return (right * xc + up * yc + forward).norm()
    }
}

/** Nominal world-camera focal length as a fraction of image width (see [HeadFrame.rayThrough]). */
const val FX_OVER_W = 0.604f

/** A sub-rectangle of a captured frame, normalized 0..1 (v down). */
data class Region(val label: String, val u0: Float, val v0: Float, val u1: Float, val v1: Float) {
    val cu get() = (u0 + u1) / 2f
    val cv get() = (v0 + v1) / 2f
    fun iou(o: Region): Float {
        val iw = (minOf(u1, o.u1) - maxOf(u0, o.u0)).coerceAtLeast(0f)
        val ih = (minOf(v1, o.v1) - maxOf(v0, o.v0)).coerceAtLeast(0f)
        val inter = iw * ih
        val union = (u1 - u0) * (v1 - v0) + (o.u1 - o.u0) * (o.v1 - o.v0) - inter
        return if (union > 0f) inter / union else 0f
    }

    companion object {
        val FULL = Region("full", 0f, 0f, 1f, 1f)

        /**
         * The 3x3 overlapping windows npu_server op 9 embeds, row-major (top-left .. bottom-right).
         * Landscape token grid 14x20 with 6x8 windows at rows {0,4,8}, cols {0,6,12}; portrait is
         * the transpose (20x14, 8x6 windows at rows {0,6,12}, cols {0,4,8}).
         */
        fun grid3x3(landscape: Boolean): List<Region> {
            val gh = if (landscape) 14 else 20
            val gw = if (landscape) 20 else 14
            val sh = if (landscape) 6 else 8
            val sw = if (landscape) 8 else 6
            val rs = intArrayOf(0, (gh - sh) / 2, gh - sh)
            val cs = intArrayOf(0, (gw - sw) / 2, gw - sw)
            val rowName = arrayOf("top", "middle", "bottom")
            val colName = arrayOf("left", "center", "right")
            val out = ArrayList<Region>(9)
            for (ri in 0 until 3) for (ci in 0 until 3) {
                out += Region(
                    "${rowName[ri]}-${colName[ci]}",
                    cs[ci].toFloat() / gw, rs[ri].toFloat() / gh,
                    (cs[ci] + sw).toFloat() / gw, (rs[ri] + sh).toFloat() / gh,
                )
            }
            return out
        }
    }
}

/** One captured ring slot: the frame, where the head was, and its embeddings. */
class PanelCapture(
    val slot: Int,
    val thumb: Bitmap,
    val head: HeadFrame,
    val aspectHOverW: Float,
    val fullEmb: FloatArray,
    val regions: List<Pair<Region, FloatArray>>,
    val visionMs: Long,
    val textMs: Long,
    val source: String,
) {
    /** The ring-panel slice (the frame's pixels covering this slot's 30 x 40 deg). */
    var slice: Bitmap? = null
}

data class Hit(
    val capture: PanelCapture,
    val region: Region,
    val score: Float,
    /** Unit ray from the capture head position through the region centre. */
    val dir: V3,
)

object PanoSearch {
    fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    /**
     * Scores every region of every capture against [query], then keeps the best non-overlapping
     * ones: within a capture, regions with IoU > 0.3 against a better hit are dropped; across
     * captures (overlapping ring slots see the same object), hits whose rays are within
     * [minSepDeg] of a better hit are dropped. Returns up to [maxHits] hits that score within
     * [margin] of the best (cell scores in one room sit within ~0.03 of each other, so a wider
     * margin mostly adds wrong cards).
     */
    fun search(
        captures: List<PanelCapture>,
        query: FloatArray,
        maxHits: Int = 3,
        margin: Float = 0.01f,
        minSepDeg: Float = 12f,
    ): List<Hit> {
        val all = ArrayList<Hit>()
        // Rank by the best grid cell, not the whole-frame embedding: on the labeled sweep this
        // found the right frame 9/10 and cell 8/10, vs 7/10 frames whole-image-first
        // (embgemma2/grid_eval/REPORT.md).
        for (c in captures) for ((r, e) in c.regions) {
            if (r == Region.FULL) continue
            all += Hit(c, r, dot(query, e), c.head.rayThrough(r.cu, r.cv, c.aspectHOverW))
        }
        all.sortByDescending { it.score }
        if (all.isEmpty()) return emptyList()
        val best = all[0].score
        val cosSep = cos(Math.toRadians(minSepDeg.toDouble())).toFloat()
        val kept = ArrayList<Hit>()
        for (h in all) {
            if (h.score < best - margin) break
            val clash = kept.any { k ->
                (k.capture === h.capture && k.region.iou(h.region) > 0.3f) || k.dir.dot(h.dir) > cosSep
            }
            if (!clash) kept += h
            if (kept.size >= maxHits) break
        }
        return kept
    }
}
