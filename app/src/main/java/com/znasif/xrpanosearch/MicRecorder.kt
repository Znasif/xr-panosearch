package com.znasif.xrpanosearch

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlin.math.sqrt

/**
 * 16 kHz mono capture with simple energy endpointing: records until [silenceMs] of quiet follows
 * speech, or [maxMs] elapses. Blocking; call off the main thread.
 */
object MicRecorder {
    @SuppressLint("MissingPermission")
    fun record(maxMs: Int = 5000, silenceMs: Int = 700, onLevel: (Float) -> Unit = {}): ShortArray {
        val sr = 16000
        val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, sr, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, sr / 5 * 2),
        )
        val out = ShortArray(sr * maxMs / 1000)
        var n = 0
        val chunk = ShortArray(sr / 50) // 20 ms
        var noise = -1f
        var heardSpeech = false
        var quietMs = 0
        try {
            rec.startRecording()
            while (n < out.size) {
                val got = rec.read(chunk, 0, minOf(chunk.size, out.size - n))
                if (got <= 0) break
                System.arraycopy(chunk, 0, out, n, got)
                n += got
                var e = 0.0
                for (i in 0 until got) e += chunk[i] * chunk[i].toDouble()
                val rms = sqrt(e / got).toFloat()
                onLevel(rms)
                if (noise < 0f) noise = rms
                val speech = rms > maxOf(3f * noise, 300f)
                if (!speech) noise = 0.95f * noise + 0.05f * rms
                if (speech) { heardSpeech = true; quietMs = 0 } else if (heardSpeech) quietMs += 20
                if (heardSpeech && quietMs >= silenceMs) break
            }
        } finally {
            runCatching { rec.stop() }
            rec.release()
        }
        Log.i(Engine.TAG, "MIC ${n * 1000 / sr} ms speech=$heardSpeech")
        return out.copyOf(n)
    }
}
