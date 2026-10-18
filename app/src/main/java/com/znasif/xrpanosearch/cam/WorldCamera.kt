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

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log

/**
 * Camera2 capture of one Galaxy XR world camera (id "0" by default: YUV_420_888 1920x1440 at
 * ~25 fps; id "2" also delivers world frames, id "1" is the avatar camera).
 *
 * Keeps only the newest frame (the previous one is closed as soon as a new one arrives), so
 * nothing queues up and frames are never copied unless [snapshot] is called. Explicit
 * [start]/[stop]. The camera service refuses apps while the headset is off-head or the app is
 * in the background (CAMERA_DISABLED / disconnect); while started, this class keeps retrying
 * every [RETRY_MS] and reports why in [status].
 *
 * Timestamps: Image.timestamp == CaptureResult.SENSOR_TIMESTAMP (start of exposure). Its clock
 * is CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE: REALTIME means
 * SystemClock.elapsedRealtimeNanos() (CLOCK_BOOTTIME); UNKNOWN is, on Qualcomm HALs,
 * CLOCK_MONOTONIC (System.nanoTime()). [timestampSource] reports it and [ageNs] converts a
 * frame time into "nanoseconds before now" with the matching clock, which Unity subtracts from
 * Time.realtimeSinceStartupAsDouble to look up the head pose at capture time.
 */
class WorldCamera private constructor(private val appContext: Context) {

    private val lock = Object()
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null

    // Guarded by lock.
    private var latest: Image? = null
    private var latestFrameNumber = 0L
    private var frameCount = 0L
    private var lastFrameElapsedMs = 0L
    private val exposureByTs = LinkedHashMap<Long, LongArray>() // ts -> [exposure, skew]
    private var latestMeanLuma = -1.0 // sampled about once a second, -1 = unknown
    private var lumaSampledMs = 0L

    @Volatile private var wanted = false
    @Volatile private var state = "STOPPED"
    @Volatile private var cameraId = DEFAULT_CAMERA_ID
    @Volatile private var width = DEFAULT_WIDTH
    @Volatile private var height = DEFAULT_HEIGHT
    @Volatile var timestampSource: String = "UNKNOWN"; private set

    val isStarted: Boolean get() = wanted

    fun hasPermission(): Boolean =
        appContext.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    /** Starts (or restarts with new settings) streaming. Returns [status]. */
    fun start(id: String = DEFAULT_CAMERA_ID, w: Int = DEFAULT_WIDTH, h: Int = DEFAULT_HEIGHT): String {
        synchronized(lock) {
            if (wanted && id == cameraId && w == width && h == height && device != null) return status()
            if (wanted) closeLocked()
            cameraId = id; width = w; height = h
            wanted = true
            if (thread == null) {
                thread = HandlerThread("WorldCam").also { it.start() }
                handler = Handler(thread!!.looper)
            }
            handler!!.post { open() }
            state = "OPENING"
        }
        return status()
    }

    fun stop() {
        synchronized(lock) {
            wanted = false
            closeLocked()
            state = "STOPPED"
        }
    }

    /**
     * "STREAMING", "OPENING", "STOPPED", "WAITING_FOR_FRAMES", or a "<CODE>: explanation" string
     * meant to be shown to the user as-is (NO_PERMISSION, DISABLED, DISCONNECTED, IN_USE, ERROR,
     * STALLED).
     */
    fun status(): String {
        val s = state
        if (s == "STREAMING") {
            val last = synchronized(lock) { lastFrameElapsedMs }
            if (last == 0L) return "WAITING_FOR_FRAMES"
            val gap = SystemClock.elapsedRealtime() - last
            if (gap > STALL_MS) return "STALLED: no camera frame for $gap ms (headset off-head or app in background?)"
            val dark = synchronized(lock) { latestMeanLuma }
            if (dark in 0.0..DARK_LUMA) return "TOO_DARK: camera image is nearly black (mean brightness %.0f/255); add light".format(dark)
        }
        return s
    }

    fun frameCount(): Long = synchronized(lock) { frameCount }

    /** Copies the newest frame (a few ms), or null if none has arrived yet. */
    fun snapshot(): YuvFrame? = synchronized(lock) {
        val img = latest ?: return null
        val meta = exposureByTs[img.timestamp]
        try {
            YuvFrame.copyOf(img, meta?.get(0) ?: 0L, meta?.get(1) ?: 0L, latestFrameNumber)
        } catch (e: IllegalStateException) { // image already closed
            null
        }
    }

    /** Nanoseconds between the camera clock time [tsNs] and now, using the matching clock. */
    fun ageNs(tsNs: Long): Long = nowNs() - tsNs

    /** "Now" in the camera timestamp clock. */
    fun nowNs(): Long = if (timestampSource == "REALTIME") SystemClock.elapsedRealtimeNanos() else System.nanoTime()

    // ---- camera thread ----

    @SuppressLint("MissingPermission")
    private fun open() {
        if (!wanted) return
        if (!hasPermission()) {
            state = "NO_PERMISSION: grant the camera permission (android.permission.CAMERA)"
            retryLater()
            return
        }
        val cm = appContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            val ch = cm.getCameraCharacteristics(cameraId)
            timestampSource = when (ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)) {
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
                else -> "UNKNOWN"
            }
            val sizes = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.getOutputSizes(ImageFormat.YUV_420_888)
            if (sizes != null && sizes.none { it.width == width && it.height == height }) {
                state = "ERROR: camera $cameraId has no YUV ${width}x$height output (has ${sizes.joinToString()})"
                return
            }
            Log.i(TAG, "Opening camera $cameraId ${width}x$height, timestamp source $timestampSource")
            cm.openCamera(cameraId, deviceCallback, handler)
        } catch (e: CameraAccessException) {
            state = describe(e)
            Log.w(TAG, "openCamera failed: $state")
            retryLater()
        } catch (e: SecurityException) {
            state = "NO_PERMISSION: ${e.message}"
            retryLater()
        } catch (e: IllegalArgumentException) {
            state = "ERROR: no camera '$cameraId' (${e.message})"
        }
    }

    private val deviceCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            synchronized(lock) {
                if (!wanted) { camera.close(); return }
                device = camera
            }
            createSession(camera)
        }

        override fun onDisconnected(camera: CameraDevice) {
            Log.w(TAG, "Camera $cameraId disconnected")
            synchronized(lock) { closeLocked() }
            state = "DISCONNECTED: camera taken by the system or another app (headset off-head?); retrying"
            retryLater()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            Log.w(TAG, "Camera $cameraId error $error")
            synchronized(lock) { closeLocked() }
            state = when (error) {
                ERROR_CAMERA_DISABLED -> "DISABLED: camera disabled by policy (wear the headset and keep the app in the foreground); retrying"
                ERROR_CAMERA_IN_USE, ERROR_MAX_CAMERAS_IN_USE -> "IN_USE: camera $cameraId is used by another app; retrying"
                else -> "ERROR: camera device error $error; retrying"
            }
            retryLater()
        }
    }

    private fun createSession(camera: CameraDevice) {
        val r = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, MAX_IMAGES)
        r.setOnImageAvailableListener({ onImage(it) }, handler)
        synchronized(lock) { reader = r }
        try {
            @Suppress("DEPRECATION")
            camera.createCaptureSession(listOf(r.surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                    synchronized(lock) {
                        if (!wanted || device !== camera) { s.close(); return }
                        session = s
                    }
                    try {
                        val rb = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                        rb.addTarget(r.surface)
                        s.setRepeatingRequest(rb.build(), captureCallback, handler)
                        state = "STREAMING"
                        Log.i(TAG, "Camera $cameraId streaming")
                    } catch (e: CameraAccessException) {
                        state = describe(e); retryLater()
                    } catch (e: IllegalStateException) {
                        state = "ERROR: ${e.message}"; retryLater()
                    }
                }

                override fun onConfigureFailed(s: CameraCaptureSession) {
                    state = "ERROR: capture session configuration failed; retrying"
                    synchronized(lock) { closeLocked() }
                    retryLater()
                }
            }, handler)
        } catch (e: CameraAccessException) {
            state = describe(e)
            synchronized(lock) { closeLocked() }
            retryLater()
        }
    }

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
            val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
            val skew = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW) ?: 0L
            synchronized(lock) {
                exposureByTs[ts] = longArrayOf(exp, skew)
                while (exposureByTs.size > 16) exposureByTs.remove(exposureByTs.keys.first())
            }
        }
    }

    private fun onImage(r: ImageReader) {
        val img = try { r.acquireLatestImage() } catch (e: IllegalStateException) { null } ?: return
        synchronized(lock) {
            if (reader !== r) { img.close(); return }
            latest?.close()
            latest = img
            frameCount++
            latestFrameNumber = frameCount
            lastFrameElapsedMs = SystemClock.elapsedRealtime()
            if (lastFrameElapsedMs - lumaSampledMs > 1000) {
                lumaSampledMs = lastFrameElapsedMs
                latestMeanLuma = sampleLuma(img)
            }
        }
    }

    /** Mean of a sparse grid (every 32nd pixel) of the Y plane, without copying the frame. */
    private fun sampleLuma(img: Image): Double = try {
        val p = img.planes[0]
        val buf = p.buffer
        var sum = 0L; var n = 0
        var yy = 0
        while (yy < img.height) {
            var xx = 0
            while (xx < img.width) { sum += buf.get(yy * p.rowStride + xx * p.pixelStride).toInt() and 0xff; n++; xx += 32 }
            yy += 32
        }
        if (n == 0) -1.0 else sum.toDouble() / n
    } catch (t: Throwable) { -1.0 }

    private fun retryLater() {
        val h = handler ?: return
        h.removeCallbacksAndMessages(RETRY_TOKEN)
        if (Build.VERSION.SDK_INT >= 28) {
            h.postDelayed({ if (wanted && device == null) open() }, RETRY_TOKEN, RETRY_MS)
        } else {
            h.postDelayed({ if (wanted && device == null) open() }, RETRY_MS)
        }
    }

    private fun closeLocked() {
        try { session?.close() } catch (_: Throwable) {}
        session = null
        try { device?.close() } catch (_: Throwable) {}
        device = null
        try { latest?.close() } catch (_: Throwable) {}
        latest = null
        try { reader?.close() } catch (_: Throwable) {}
        reader = null
        lastFrameElapsedMs = 0L
        latestMeanLuma = -1.0
        exposureByTs.clear()
    }

    private fun describe(e: CameraAccessException): String = when (e.reason) {
        CameraAccessException.CAMERA_DISABLED -> "DISABLED: camera disabled by policy (wear the headset and keep the app in the foreground); retrying"
        CameraAccessException.CAMERA_DISCONNECTED -> "DISCONNECTED: camera unavailable (headset off-head?); retrying"
        CameraAccessException.CAMERA_IN_USE, CameraAccessException.MAX_CAMERAS_IN_USE -> "IN_USE: camera in use by another app; retrying"
        else -> "ERROR: camera access ${e.reason}: ${e.message}; retrying"
    }

    companion object {
        private const val TAG = "WorldCam"
        const val DEFAULT_CAMERA_ID = "0"
        const val DEFAULT_WIDTH = 1920
        const val DEFAULT_HEIGHT = 1440
        private const val MAX_IMAGES = 3
        private const val RETRY_MS = 2000L
        private const val STALL_MS = 1500L
        private const val DARK_LUMA = 12.0
        private val RETRY_TOKEN = Any()

        @Volatile private var instance: WorldCamera? = null

        @JvmStatic
        fun get(context: Context): WorldCamera = instance ?: synchronized(this) {
            instance ?: WorldCamera(context.applicationContext ?: context).also { instance = it }
        }
    }
}
