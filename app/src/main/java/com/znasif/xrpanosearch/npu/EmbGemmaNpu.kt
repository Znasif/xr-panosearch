package com.znasif.xrpanosearch.npu

import android.content.Context
import android.system.Os
import android.util.Log
import java.io.File

/**
 * EmbeddingGemma 2 on the Hexagon HTP, in this process (libembgemma_npu.so). Project Aura lets
 * apps open the CDSP directly, so there is no Shizuku/npu_server hop: every call is a JNI call
 * straight into ExecuTorch + QNN. Not thread-safe: call everything from one worker thread.
 *
 * Models and CPU assets live in [modelDir] (app-specific external storage, pushed by
 * tools/setup_device.sh).
 */
class EmbGemmaNpu(context: Context) {
    companion object {
        const val TAG = "EmbGemmaNpu"
        const val DIM = 768
        /** Weight-shared text .pte with right-sized buckets (q32, a160, img288, s512, grid 9x64). */
        const val TEXT_MULTI = "embgemma2_text_multi_qnn_fp16.pte"
        /** Older layout: one text .pte per bucket. Used only when [TEXT_MULTI] is absent. */
        val TEXT_SINGLE = listOf("embgemma2_text_S128_qnn_fp16.pte", "embgemma2_text_S512_qnn_fp16.pte")
        val OTHER_MODELS = listOf("embgemma2_vision_landscape_qnn_fp16.pte", "embgemma2_audio_5s_qnn_fp16.pte")
        val CPU_ASSETS = listOf(
            "tokenizer.json",
            "embed_tokens_262144x512_f16.bin",
            "gemma4_mel_filters_128x257.f32",
        )

        @JvmStatic private external fun nativeInit(dir: String, perfMode: Int)
        @JvmStatic private external fun nativePreload(file: String): Long
        @JvmStatic private external fun nativeText(text: String, times: LongArray): FloatArray
        @JvmStatic private external fun nativeImageGrid(pixels: FloatArray, includeGrid: Boolean, times: LongArray): FloatArray
        @JvmStatic private external fun nativeAudio(pcm: ShortArray, times: LongArray): FloatArray
    }

    val modelDir: File = File(context.getExternalFilesDir(null), "embgemma2")
    private val nativeDir = context.applicationInfo.nativeLibraryDir
    private var ready = false

    /**
     * Loads the library, CPU assets and all models (a few seconds cold). Returns "READY" or an
     * error string for the UI. [perfMode] overrides the HTP performance mode (-1 = AOT default).
     */
    fun init(perfMode: Int = -1): String {
        if (ready) return "READY"
        val models = (if (File(modelDir, TEXT_MULTI).isFile) listOf(TEXT_MULTI) else TEXT_SINGLE) + OTHER_MODELS
        val missing = (models + CPU_ASSETS).filterNot { File(modelDir, it).isFile }
        if (missing.isNotEmpty()) return "MISSING in $modelDir: ${missing.joinToString()} (run tools/setup_device.sh)"
        return try {
            // The CDSP loads libQnnHtpV79Skel.so from ADSP_LIBRARY_PATH; it must be set before
            // the QNN backend initialises.
            Os.setenv("ADSP_LIBRARY_PATH", nativeDir, true)
            System.loadLibrary("embgemma_npu")
            nativeInit(modelDir.absolutePath, perfMode)
            for (m in models) Log.i(TAG, "preload $m: ${nativePreload(m) / 1000} ms")
            ready = true
            "READY"
        } catch (t: Throwable) {
            Log.e(TAG, "init failed", t)
            "ERROR: ${t.message}"
        }
    }

    class Text(val embedding: FloatArray, val nTokens: Int, val textUs: Long, val wallUs: Long)
    class ImageGrid(val full: FloatArray, val grid: List<FloatArray>, val visionUs: Long, val textUs: Long, val wallUs: Long)
    class Audio(val embedding: FloatArray, val nSoftTokens: Int, val melUs: Long, val audioUs: Long, val textUs: Long, val wallUs: Long)

    /** [text] must already carry the task prefix. */
    fun text(text: String): Text {
        val t0 = System.nanoTime()
        val times = LongArray(2)
        val e = nativeText(text, times)
        return Text(e, times[1].toInt(), times[0], (System.nanoTime() - t0) / 1000)
    }

    /** [pixels]: CHW f32 [3, 672, 960] in [0, 1] (ImagePrep). */
    fun imageGrid(pixels: FloatArray, includeGrid: Boolean = true): ImageGrid {
        val t0 = System.nanoTime()
        val times = LongArray(2)
        val out = nativeImageGrid(pixels, includeGrid, times)
        val embs = (0 until out.size / DIM).map { out.copyOfRange(it * DIM, (it + 1) * DIM) }
        return ImageGrid(embs[0], embs.drop(1), times[0], times[1], (System.nanoTime() - t0) / 1000)
    }

    /** 16 kHz mono PCM; only the first 5 s are used. */
    fun audio(pcm: ShortArray): Audio {
        val t0 = System.nanoTime()
        val times = LongArray(4)
        val e = nativeAudio(pcm, times)
        return Audio(e, times[3].toInt(), times[0], times[1], times[2], (System.nanoTime() - t0) / 1000)
    }
}
