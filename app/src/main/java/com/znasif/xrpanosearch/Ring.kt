package com.znasif.xrpanosearch

import com.znasif.xrpanosearch.cam.CropRect
import kotlin.math.tan

/**
 * Panorama ring geometry, after Project_Aura (D:\Projects\Project_Aura, MainActivity.kt
 * computeRingGeometry): [SLOTS] flat panels tangent to a circle of [RADIUS_M] around the head at
 * eye level. A panel 2R*tan(step/2) wide spans exactly one step as seen from the centre, so
 * neighbours meet edge to edge; each shows only the camera pixels covering its own STEP x
 * [DISPLAY_VFOV_DEG] of the world, which makes the ring read as one continuous, world-aligned
 * panorama. The camera's full frame (~79 x 64 deg) is still what gets embedded.
 */
object Ring {
    const val SLOTS = 12
    const val STEP_DEG = 360f / SLOTS
    const val RADIUS_M = 1.0f
    /** Vertical slice shown per panel; the camera sees ~64 deg, 40 keeps panels a comfortable height. */
    const val DISPLAY_VFOV_DEG = 40f

    private fun tanDeg(d: Float) = tan(Math.toRadians(d.toDouble())).toFloat()

    val panelWidthM = 2f * RADIUS_M * tanDeg(STEP_DEG / 2f)       // 0.536 m
    val panelHeightM = 2f * RADIUS_M * tanDeg(DISPLAY_VFOV_DEG / 2f) // 0.728 m

    /** Ring frame for slot [slot] (forward = slot direction, level). */
    fun slotFrame(origin: HeadFrame, slot: Int): HeadFrame =
        HeadFrame.fromYawPitch(origin.pos, origin.yawDeg + slot * STEP_DEG, 0f)

    /** Centre of slot [slot]'s panel, at the origin's eye height. */
    fun slotCentre(origin: HeadFrame, slot: Int): V3 = origin.pos + slotFrame(origin, slot).forward * RADIUS_M

    /**
     * The crop of a [w]x[h] camera frame (taken with head pose [head]) that covers slot [slot]'s
     * panel exactly: projects the panel centre through the nominal pinhole, then takes
     * 2f*tan(step/2) x 2f*tan(vfov/2) pixels around it. Returns null when the slice falls
     * mostly outside the frame (head turned too far from the slot).
     */
    fun cropFor(origin: HeadFrame, slot: Int, head: HeadFrame, w: Int, h: Int): CropRect? {
        val d = (slotCentre(origin, slot) - head.pos).norm()
        val z = d.dot(head.forward)
        if (z < 0.5f) return null
        val f = FX_OVER_W * w
        val cx = w / 2f + f * d.dot(head.right) / z
        val cy = h / 2f - f * d.dot(head.up) / z
        val cw = (2f * f * tanDeg(STEP_DEG / 2f)).toInt()
        val ch = (2f * f * tanDeg(DISPLAY_VFOV_DEG / 2f)).toInt().coerceAtMost(h)
        val x0 = (cx - cw / 2f).toInt()
        val y0 = (cy - ch / 2f).toInt()
        // Allow a little slop at the frame edge, then clamp.
        if (x0 < -cw / 4 || x0 + cw > w + cw / 4 || y0 < -ch / 4 || y0 + ch > h + ch / 4) return null
        return CropRect(x0.coerceIn(0, w - cw), y0.coerceIn(0, h - ch), cw, ch)
    }

    /** Signed yaw of [head] relative to [origin], in (-180, 180]. */
    fun relYaw(origin: HeadFrame, head: HeadFrame): Float {
        var d = (head.yawDeg - origin.yawDeg) % 360f
        if (d > 180f) d -= 360f
        if (d <= -180f) d += 360f
        return d
    }
}
