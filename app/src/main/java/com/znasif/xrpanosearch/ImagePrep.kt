package com.znasif.xrpanosearch

import android.graphics.Bitmap
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Bitmap -> EmbeddingGemma 2 vision input: stretch to 960x672 (landscape) or 672x960 (portrait),
 * planar CHW f32 in [0,1]. Uses PIL-style antialiased bicubic (a = -0.5), horizontal pass then
 * vertical, each rounded to 8 bits like PIL. The vision port measured that bilinear or area
 * resizing costs up to 0.035 embedding cosine on thin text (embgemma2/vision/REPORT.md).
 */
object ImagePrep {
    const val LONG = 960
    const val SHORT = 672

    private val nThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 6)
    private val pool = Executors.newFixedThreadPool(nThreads) { r -> Thread(r, "imageprep").apply { isDaemon = true } }

    /** Runs [body] over [0, n) split into one contiguous chunk per pool thread. */
    private fun parallel(n: Int, body: (Int, Int) -> Unit) {
        val step = (n + nThreads - 1) / nThreads
        pool.invokeAll((0 until n step step).map { a -> Callable { body(a, minOf(n, a + step)) } })
            .forEach { it.get() }
    }

    /** Returns (pixels, orient) where orient 0 = landscape, 1 = portrait (NpuClient convention). */
    fun toChw(src: Bitmap): Pair<FloatArray, Int> {
        val landscape = src.width >= src.height
        val ow = if (landscape) LONG else SHORT
        val oh = if (landscape) SHORT else LONG
        val w = src.width
        val h = src.height
        val argb = IntArray(w * h)
        src.getPixels(argb, 0, w, 0, 0, w, h)

        // Horizontal pass: [h][ow] per channel, 8-bit rounded.
        val hx = coeffs(w, ow)
        val tmp = Array(3) { FloatArray(h * ow) }
        parallel(h) { ya, yb -> for (y in ya until yb) {
            val row = y * w
            for (x in 0 until ow) {
                val x0 = hx.bounds[2 * x]
                val n = hx.bounds[2 * x + 1]
                val k = x * hx.ksize
                var r = 0f; var g = 0f; var b = 0f
                for (i in 0 until n) {
                    val p = argb[row + x0 + i]
                    val c = hx.kk[k + i]
                    r += c * ((p shr 16) and 0xFF)
                    g += c * ((p shr 8) and 0xFF)
                    b += c * (p and 0xFF)
                }
                val o = y * ow + x
                tmp[0][o] = clamp8(r); tmp[1][o] = clamp8(g); tmp[2][o] = clamp8(b)
            }
        } }

        // Vertical pass into CHW.
        val vy = coeffs(h, oh)
        val out = FloatArray(3 * oh * ow)
        val plane = oh * ow
        parallel(3 * oh) { ra, rb -> for (row in ra until rb) {
            val c = row / oh
            val y = row % oh
            val t = tmp[c]
            val y0 = vy.bounds[2 * y]
            val n = vy.bounds[2 * y + 1]
            val k = y * vy.ksize
            for (x in 0 until ow) {
                var s = 0f
                for (i in 0 until n) s += vy.kk[k + i] * t[(y0 + i) * ow + x]
                out[c * plane + y * ow + x] = clamp8(s) / 255f
            }
        } }
        return out to (if (landscape) 0 else 1)
    }

    private class Coeffs(val ksize: Int, val bounds: IntArray, val kk: FloatArray)

    private fun bicubic(x0: Double): Double {
        val a = -0.5
        val x = if (x0 < 0) -x0 else x0
        return when {
            x < 1.0 -> ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            x < 2.0 -> (((x - 5.0) * x + 8.0) * x - 4.0) * a
            else -> 0.0
        }
    }

    /** Same construction as PIL's precompute_coeffs (support 2, scaled up when downsampling). */
    private fun coeffs(inSize: Int, outSize: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = max(scale, 1.0)
        val support = 2.0 * filterScale
        val ksize = ceil(support).toInt() * 2 + 1
        val bounds = IntArray(outSize * 2)
        val kk = FloatArray(outSize * ksize)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val ss = 1.0 / filterScale
            val xmin = max((center - support + 0.5).toInt(), 0)
            val xmax = min((center + support + 0.5).toInt(), inSize) - xmin
            var total = 0.0
            val ws = DoubleArray(xmax)
            for (x in 0 until xmax) {
                val w = bicubic((x + xmin - center + 0.5) * ss)
                ws[x] = w
                total += w
            }
            for (x in 0 until xmax) kk[xx * ksize + x] = if (total != 0.0) (ws[x] / total).toFloat() else 0f
            bounds[2 * xx] = xmin
            bounds[2 * xx + 1] = xmax
        }
        return Coeffs(ksize, bounds, kk)
    }

    private fun clamp8(v: Float): Float = v.roundToInt().coerceIn(0, 255).toFloat()
}
