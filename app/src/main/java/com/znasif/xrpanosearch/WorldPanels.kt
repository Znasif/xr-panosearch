package com.znasif.xrpanosearch

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.xr.runtime.Session
import androidx.xr.runtime.math.FloatSize2d
import androidx.xr.runtime.math.Pose
import androidx.xr.runtime.math.Quaternion
import androidx.xr.runtime.math.Vector3
import androidx.xr.scenecore.PanelEntity
import androidx.xr.scenecore.scene

/**
 * World-locked panels (SceneCore PanelEntity, parented to the activity space so poses are explicit;
 * the create() overloads without a parent leave the entity detached and it is never drawn):
 * the panorama ring ([Ring] geometry), the control panel, and the result cards. Main thread only.
 */
class WorldPanels(private val ctx: Context) {
    private class Cell(
        val entity: PanelEntity, val frame: FrameLayout, val image: ImageView,
        val label: TextView, val bar: View, var shown: Bitmap?,
    )

    private val cells = HashMap<Int, Cell>()
    private val cards = ArrayList<PanelEntity>()
    private var control: PanelEntity? = null
    private var controlYaw = 0f

    companion object {
        /** Cards float just inside the ring surface, along the hit's camera ray. */
        const val CARD_DIST = 0.85f
        const val CARD_W = 0.24f
        const val CONTROL_W = 0.62f
        const val CONTROL_H = 0.44f
        const val CONTROL_DIST = 0.6f
        const val CONTROL_DROP = 0.42f
        const val CONTROL_TILT_DEG = 40f
        /** Re-centre the control panel once the head has turned this far from it. */
        const val CONTROL_FOLLOW_DEG = 45f
    }

    private fun yawQ(yawDeg: Float) = Quaternion.fromEulerAngles(0f, yawDeg, 0f)

    /** Panel facing back along -[dirToPanel] (a panel's front is its +Z). */
    private fun facing(pos: V3, dirToPanel: V3): Pose {
        val yaw = Math.toDegrees(kotlin.math.atan2(-dirToPanel.x, -dirToPanel.z).toDouble()).toFloat()
        return Pose(Vector3(pos.x, pos.y, pos.z), yawQ(yaw))
    }

    private fun box(bg: Int, radius: Float, stroke: Int = 0, strokeColor: Int = 0) = GradientDrawable().apply {
        setColor(bg); cornerRadius = radius
        if (stroke > 0) setStroke(stroke, strokeColor)
    }

    // ------------------------------------------------------------------ ring

    /** Builds (or moves) the 12 ring panels around [origin]. */
    fun buildRing(session: Session, origin: HeadFrame) {
        for (slot in 0 until Ring.SLOTS) {
            val sf = Ring.slotFrame(origin, slot)
            val pose = facing(Ring.slotCentre(origin, slot), sf.forward)
            val cell = cells.getOrPut(slot) {
                val image = ImageView(ctx).apply { scaleType = ImageView.ScaleType.FIT_XY }
                val label = TextView(ctx).apply {
                    setTextColor(Color.WHITE); textSize = 26f; gravity = Gravity.CENTER
                }
                val bar = View(ctx).apply { setBackgroundColor(0xFF4FC3F7.toInt()); pivotX = 0f; scaleX = 0f }
                val frame = FrameLayout(ctx).apply {
                    addView(image, FrameLayout.LayoutParams(-1, -1))
                    addView(label, FrameLayout.LayoutParams(-1, -1))
                    addView(bar, FrameLayout.LayoutParams(-1, 18, Gravity.BOTTOM))
                }
                val e = PanelEntity.create(
                    session, frame, FloatSize2d(Ring.panelWidthM, Ring.panelHeightM), "ring$slot", pose,
                    session.scene.activitySpace,
                )
                e.cornerRadius = 0f
                Cell(e, frame, image, label, bar, null)
            }
            cell.entity.setPose(pose)
            setCell(slot, null, "${(slot * Ring.STEP_DEG).toInt()}°", active = false, progress = 0f)
        }
        Log.i(Engine.TAG, "ring built: ${Ring.SLOTS} x %.3f x %.3f m at r=%.2f, origin ${origin.pos} yaw %.1f"
            .format(Ring.panelWidthM, Ring.panelHeightM, Ring.RADIUS_M, origin.yawDeg))
    }

    /**
     * Updates one ring panel: [image] (live view or captured slice, null = empty), a caption, the
     * active highlight, and the capture countdown bar (0..1).
     */
    fun setCell(slot: Int, image: Bitmap?, text: String, active: Boolean, progress: Float) {
        val c = cells[slot] ?: return
        if (image !== c.shown) { c.shown = image; c.image.setImageBitmap(image) }
        c.label.text = text
        c.frame.background = if (image == null) box(if (active) 0x99004D73.toInt() else 0x66263238, 0f) else null
        // Empty slots get a faint outline so the whole ring is visible as a guide.
        c.frame.foreground = when {
            active -> box(0x00000000, 0f, 8, 0xFF4FC3F7.toInt())
            image == null -> box(0x00000000, 0f, 3, 0x88FFFFFF.toInt())
            else -> null
        }
        c.bar.scaleX = progress.coerceIn(0f, 1f)
    }

    // ------------------------------------------------------------------ control panel

    /** Hosts [view] as the control panel, low and in front of [head], tilted up toward the eyes. */
    fun showControl(session: Session, view: View, head: HeadFrame) {
        val e = control ?: PanelEntity.create(
            session, view, FloatSize2d(CONTROL_W, CONTROL_H), "control", Pose(), session.scene.activitySpace,
        ).also { control = it }
        placeControl(e, head)
    }

    /** Keeps the control panel within reach: re-centres it when the head turns away. */
    fun followControl(head: HeadFrame) {
        val e = control ?: return
        var d = (head.yawDeg - controlYaw) % 360f
        if (d > 180f) d -= 360f
        if (d < -180f) d += 360f
        if (kotlin.math.abs(d) > CONTROL_FOLLOW_DEG) placeControl(e, head)
    }

    private fun placeControl(e: PanelEntity, head: HeadFrame) {
        val f = V3(head.forward.x, 0f, head.forward.z).norm()
        val pos = head.pos + f * CONTROL_DIST - V3(0f, CONTROL_DROP, 0f)
        controlYaw = head.yawDeg
        // Yaw to face the user, then pitch the top edge away so the face tilts up toward the eyes.
        val q = yawQ(head.yawDeg) * Quaternion.fromEulerAngles(-CONTROL_TILT_DEG, 0f, 0f)
        e.setPose(Pose(Vector3(pos.x, pos.y, pos.z), q))
    }

    // ------------------------------------------------------------------ result cards

    fun clearCards() {
        cards.forEach { it.dispose() }
        cards.clear()
    }

    /** One card per hit, [CARD_DIST] along the hit's camera ray (just inside the ring), facing the user. */
    fun showHits(session: Session, hits: List<Hit>) {
        clearCards()
        hits.forEachIndexed { i, h ->
            val pos = h.capture.head.pos + h.dir * CARD_DIST
            val title = TextView(ctx).apply {
                text = "#${i + 1}  %.2f".format(h.score)
                setTextColor(Color.WHITE); textSize = 20f; setTypeface(typeface, Typeface.BOLD)
            }
            val img = ImageView(ctx).apply { setImageBitmap(crop(h)); scaleType = ImageView.ScaleType.FIT_CENTER }
            val root = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                background = box(0xEE0D47A1.toInt(), 24f, 6, 0xFFFFD54F.toInt())
                setPadding(16, 10, 16, 10)
                addView(title)
                addView(img, LinearLayout.LayoutParams(-1, 0, 1f))
            }
            val e = PanelEntity.create(
                session, root, FloatSize2d(CARD_W, CARD_W * 0.85f), "hit$i", facing(pos, h.dir),
                session.scene.activitySpace,
            )
            Log.i(Engine.TAG, "CARD #${i + 1} ${h.capture.source}/${h.region.label} at $pos")
            cards += e
        }
    }

    private fun crop(h: Hit): Bitmap {
        val t = h.capture.thumb
        val r = h.region
        val x = (r.u0 * t.width).toInt().coerceIn(0, t.width - 1)
        val y = (r.v0 * t.height).toInt().coerceIn(0, t.height - 1)
        val w = ((r.u1 - r.u0) * t.width).toInt().coerceIn(1, t.width - x)
        val hh = ((r.v1 - r.v0) * t.height).toInt().coerceIn(1, t.height - y)
        return Bitmap.createBitmap(t, x, y, w, hh)
    }

    fun disposeAll() {
        clearCards()
        cells.values.forEach { it.entity.dispose() }
        cells.clear()
        control?.dispose(); control = null
    }
}
