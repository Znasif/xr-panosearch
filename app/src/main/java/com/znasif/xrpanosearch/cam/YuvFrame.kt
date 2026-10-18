/*
 * Copyright 2025 The Embardiment Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.znasif.xrpanosearch.cam

import android.graphics.Bitmap
import android.media.Image
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * An owned copy of one YUV_420_888 camera frame (tightly packed Y, U and V planes, chroma at
 * half resolution). Only ever kept in memory; never written to storage unless the app opts in
 * to debug dumps ([WorldCamBridge.setDebugDumps]).
 */
class YuvFrame(
    val width: Int,
    val height: Int,
    val y: ByteArray,
    val u: ByteArray,
    val v: ByteArray,
    /** Camera2 SENSOR_TIMESTAMP (start of exposure), ns, in the clock named by [WorldCamera.timestampSource]. */
    val timestampNs: Long,
    /** Exposure time (ns) if the capture result was seen, else 0. */
    val exposureNs: Long,
    /** Rolling-shutter readout time (ns) if reported, else 0. */
    val rollingShutterSkewNs: Long,
    val frameNumber: Long,
) {
    /** Middle of the exposure of the middle row: the best single "capture time" for pose lookup. */
    val midExposureNs: Long get() = timestampNs + exposureNs / 2 + rollingShutterSkewNs / 2

    /** ARGB_8888 bitmap of the crop [rect] (full-range BT.601 / JFIF YUV, as Camera2 delivers). */
    fun toBitmap(rect: CropRect): Bitmap {
        val px = IntArray(rect.w * rect.h)
        val cw = width / 2
        for (j in 0 until rect.h) {
            val yy = rect.y + j
            val yRow = yy * width
            val cRow = (yy shr 1) * cw
            val o = j * rect.w
            for (i in 0 until rect.w) {
                val xx = rect.x + i
                val yv = (y[yRow + xx].toInt() and 0xff).toFloat()
                val ci = cRow + (xx shr 1)
                val uu = (u[ci].toInt() and 0xff) - 128f
                val vv = (v[ci].toInt() and 0xff) - 128f
                val r = clamp(yv + 1.402f * vv)
                val g = clamp(yv - 0.344136f * uu - 0.714136f * vv)
                val b = clamp(yv + 1.772f * uu)
                px[o + i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(px, rect.w, rect.h, Bitmap.Config.ARGB_8888)
    }

    /** Mean luma (0-255) of [rect], sampled every [step] pixels: a cheap "is it too dark" check. */
    fun meanLuma(rect: CropRect = CropRect(0, 0, width, height), step: Int = 8): Double {
        var sum = 0L; var n = 0
        var j = rect.y
        while (j < rect.y + rect.h) {
            var i = rect.x
            val row = j * width
            while (i < rect.x + rect.w) { sum += y[row + i].toInt() and 0xff; n++; i += step }
            j += step
        }
        return if (n == 0) 0.0 else sum.toDouble() / n
    }

    private fun clamp(f: Float): Int = min(255, max(0, f.roundToInt()))

    companion object {
        /** Copies an Image's planes (any row/pixel stride) into a packed frame. */
        fun copyOf(img: Image, exposureNs: Long, skewNs: Long, frameNumber: Long): YuvFrame {
            val w = img.width; val h = img.height
            val planes = img.planes
            val y = ByteArray(w * h)
            val u = ByteArray((w / 2) * (h / 2))
            val v = ByteArray((w / 2) * (h / 2))
            copyPlane(planes[0], w, h, y)
            copyPlane(planes[1], w / 2, h / 2, u)
            copyPlane(planes[2], w / 2, h / 2, v)
            return YuvFrame(w, h, y, u, v, img.timestamp, exposureNs, skewNs, frameNumber)
        }

        private fun copyPlane(plane: Image.Plane, w: Int, h: Int, out: ByteArray) {
            val buf = plane.buffer.duplicate()
            val rs = plane.rowStride; val ps = plane.pixelStride
            if (ps == 1) {
                if (rs == w) {
                    buf.position(0); buf.get(out, 0, w * h)
                } else {
                    for (r in 0 until h) { buf.position(r * rs); buf.get(out, r * w, w) }
                }
                return
            }
            // Interleaved chroma (NV12/NV21 layout behind YUV_420_888): copy each row, then pick.
            val row = ByteArray(rs)
            for (r in 0 until h) {
                val start = r * rs
                val len = min(rs, buf.limit() - start)
                buf.position(start); buf.get(row, 0, len)
                val o = r * w
                for (c in 0 until w) out[o + c] = row[c * ps]
            }
        }
    }
}

/** A crop rectangle in full-frame pixels. */
data class CropRect(val x: Int, val y: Int, val w: Int, val h: Int) {
    companion object {
        /**
         * A [w] x [h] rectangle centred on (cu, cv), shifted (not shrunk) to stay inside a
         * [frameW] x [frameH] frame; only shrunk if it is larger than the frame. Even origin so
         * chroma lines up.
         */
        fun around(cu: Double, cv: Double, w: Int, h: Int, frameW: Int, frameH: Int): CropRect {
            val cw = min(max(w, 16), frameW) and 1.inv()
            val ch = min(max(h, 16), frameH) and 1.inv()
            var x = (cu - cw / 2.0).roundToInt()
            var y = (cv - ch / 2.0).roundToInt()
            x = min(max(x, 0), frameW - cw) and 1.inv()
            y = min(max(y, 0), frameH - ch) and 1.inv()
            return CropRect(x, y, cw, ch)
        }
    }
}
