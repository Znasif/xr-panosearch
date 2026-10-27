// EmbeddingGemma 2 / Gemma-4 audio log-mel front end (matches embgemma2/audio/scripts/mel_ref.py):
// 16 kHz mono s16 PCM -> float32 (pcm / 32768.0f), truncated to max_samples(T) = T*160 + 160 ->
// prepend 160 zeros ("semicausal" padding) -> frame 320, hop 160, periodic Hann ->
// 512-pt rfft magnitude |X| (257 bins) -> HTK mel filters [128, 257] -> log(mel + 1e-3) ->
// zero out frames >= n_valid. Output mel [T, 128] row-major, mask [T] (1/0), and n_tokens.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace embgemma_mel {

constexpr int kSampleRate = 16000;
constexpr int kFrame = 320;
constexpr int kHop = 160;
constexpr int kNfft = 512;
constexpr int kBins = kNfft / 2 + 1;  // 257
constexpr int kMels = 128;
constexpr double kMelFloor = 1e-3;

using cpx = std::complex<double>;

// In-place radix-2 Cooley-Tukey FFT for N = 512.
inline void fft512(cpx* a, const std::vector<cpx>& tw) {
  for (int i = 1, j = 0; i < kNfft; ++i) {
    int bit = kNfft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= kNfft; len <<= 1) {
    int half = len >> 1;
    int step = kNfft / len;
    for (int i = 0; i < kNfft; i += len) {
      for (int j = 0; j < half; ++j) {
        cpx u = a[i + j];
        cpx v = a[i + j + half] * tw[j * step];
        a[i + j] = u + v;
        a[i + j + half] = u - v;
      }
    }
  }
}

struct MelOutput {
  std::vector<float> mel;   // [T, 128]
  std::vector<float> mask;  // [T]
  int n_valid = 0;
  int n_tokens = 0;
};

// filters: [128, 257] row-major f32 from gemma4_mel_filters_128x257.f32.
inline MelOutput compute(const int16_t* pcm16, size_t n_samples, int T, const float* filters) {
  const size_t max_samp = static_cast<size_t>(T) * kHop + kHop;
  const size_t n_real = std::min(n_samples, max_samp);
  const int n_valid =
      (n_real < static_cast<size_t>(kHop + 1))
          ? 0
          : std::min<int>(static_cast<int>((n_real - (kHop + 1)) / kHop + 1), T);
  const int n_tokens = (n_valid + 3) / 4;

  const size_t buf_len = static_cast<size_t>(T - 1) * kHop + kFrame + 1;
  std::vector<float> buf(buf_len, 0.f);
  const size_t m = std::min(n_real, buf_len - kHop);
  for (size_t i = 0; i < m; ++i) {
    buf[kHop + i] = static_cast<float>(pcm16[i]) / 32768.0f;
  }

  std::vector<float> win(kFrame);
  for (int n = 0; n < kFrame; ++n) {
    win[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * n / kFrame));
  }
  std::vector<cpx> tw(kNfft / 2);
  for (int j = 0; j < kNfft / 2; ++j) {
    tw[j] = std::polar(1.0, -2.0 * M_PI * j / kNfft);
  }

  MelOutput out;
  out.mel.assign(static_cast<size_t>(T) * kMels, 0.f);
  out.mask.assign(static_cast<size_t>(T), 0.f);
  out.n_valid = n_valid;
  out.n_tokens = n_tokens;

  std::vector<cpx> spec(kNfft);
  double mag[kBins];
  for (int t = 0; t < n_valid; ++t) {
    out.mask[t] = 1.0f;
    const float* src = buf.data() + static_cast<size_t>(t) * kHop;
    for (int n = 0; n < kFrame; ++n) spec[n] = static_cast<double>(src[n] * win[n]);
    for (int n = kFrame; n < kNfft; ++n) spec[n] = 0.0;
    fft512(spec.data(), tw);
    for (int k = 0; k < kBins; ++k) mag[k] = std::abs(spec[k]);

    float* row = out.mel.data() + static_cast<size_t>(t) * kMels;
    for (int mel_idx = 0; mel_idx < kMels; ++mel_idx) {
      const float* frow = filters + static_cast<size_t>(mel_idx) * kBins;
      double sum = 0.0;
      for (int k = 0; k < kBins; ++k) sum += mag[k] * static_cast<double>(frow[k]);
      row[mel_idx] = static_cast<float>(std::log(sum + kMelFloor));
    }
  }
  return out;
}

}  // namespace embgemma_mel
