// In-process EmbeddingGemma 2 on the Hexagon HTP for Project Aura (SXR2330P, V79).
//
// Aura apps can open the CDSP directly (libcdsprpc.so is public, unsigned PD allowed), so the
// Galaxy XR route (Shizuku UserService -> npu_server child process -> stdin/stdout frames) is
// replaced by this JNI library. The model logic is npu_server's ops 8-10, unchanged:
// - Text:  HFTokenizer (tokenizer.json) + mmap'd embed_tokens_262144x512_f16.bin (* sqrt(512))
//          -> embgemma2_text_S{128,512}_qnn_fp16.pte -> [1, 768] f32.
// - Image: embgemma2_vision_landscape_qnn_fp16.pte ([1,3,672,960] -> [1,280,512]) -> full image
//          embedding via text_S512 + optional 3x3 overlapping sub-region grid
//          (9 x 48 soft tokens via text_S128) -> 1 or 10 x [768] f32.
// - Audio: embgemma_mel (16 kHz s16 PCM -> [1,500,128] mel + [1,500] mask) ->
//          embgemma2_audio_5s_qnn_fp16.pte -> [1,125,512] -> text_S128 -> [1,768] f32.
// All entry points are synchronous and must be called from one thread (the app's NPU worker).
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <pytorch/tokenizers/hf_tokenizer.h>

#include "embgemma_mel.h"

#include <android/log.h>
#include <fcntl.h>
#include <jni.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EmbGemmaNpu", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "EmbGemmaNpu", __VA_ARGS__)

constexpr size_t kEgVocab = 262144;
constexpr int kEgHidden = 512;
constexpr int kEgOutDim = 768;
constexpr float kEgEmbedScale = 22.627416610717773f;  // float32 sqrt(512)
constexpr uint32_t kEgEos = 1, kEgBos = 2;
constexpr uint32_t kEgBoi = 255999, kEgEoi = 258882;
constexpr uint32_t kEgBoa = 256000, kEgEoa = 258883;
constexpr int kVisH = 672, kVisW = 960, kVisTokens = 280;
constexpr int kAudT = 500, kAudN = 125;

uint64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

// One text-tower graph: [B,S,512] embeds + [B,S] mask -> [B,768]. Either one .pte per bucket
// (method "forward") or one weight-shared multi-method .pte (export_text_multi.py).
struct TextBucket {
  std::string file, method;
  int B, S;
};

constexpr const char* kTextMulti = "embgemma2_text_multi_qnn_fp16.pte";

struct Engine {
  std::string dir;
  std::map<std::string, std::unique_ptr<Module>> models;
  std::vector<TextBucket> buckets;  // B == 1 buckets ascending by S, then batched ones
  std::unique_ptr<tokenizers::HFTokenizer> tok;
  int embed_fd = -1;
  size_t embed_bytes = 0;
  const __fp16* embed_table = nullptr;  // mmap'd [262144, 512] fp16
  std::vector<float> mel_filters;       // [128, 257] f32
  std::vector<EValue> keep;             // last outputs; alias module memory until the next forward

  ~Engine() {
    models.clear();
    if (embed_table) munmap(const_cast<__fp16*>(embed_table), embed_bytes);
    if (embed_fd >= 0) ::close(embed_fd);
  }

  // Loads [file] (once) and its [method] (once); "" loads every method in the file.
  Module& model(const std::string& file, const std::string& method = "forward") {
    auto& m = models[file];
    if (!m) m = std::make_unique<Module>(dir + "/" + file, Module::LoadMode::MmapUseMlockIgnoreErrors);
    std::vector<std::string> names;
    if (method.empty()) {
      auto all = m->method_names();
      if (!all.ok()) throw std::runtime_error("method_names failed for " + file);
      names.assign(all->begin(), all->end());
    } else {
      names.push_back(method);
    }
    for (const auto& name : names) {
      if (m->is_method_loaded(name)) continue;
      uint64_t t0 = now_us();
      if (m->load_method(name) != Error::Ok) {
        throw std::runtime_error("load_method(" + name + ") failed for " + file);
      }
      LOGI("loaded %s:%s in %.1f ms", file.c_str(), name.c_str(), (now_us() - t0) / 1000.0);
    }
    return *m;
  }

  const executorch::aten::Tensor& run(const std::string& file, const std::string& method,
                                      const std::vector<EValue>& in, const char* what, uint64_t& us) {
    Module& m = model(file, method);
    uint64_t t0 = now_us();
    auto r = m.execute(method, in);
    us += now_us() - t0;
    if (!r.ok()) {
      throw std::runtime_error(std::string(what) + " forward failed: error " +
                               std::to_string(static_cast<int>(r.error())));
    }
    keep = std::move(*r);
    if (keep.empty() || !keep[0].isTensor()) throw std::runtime_error(std::string(what) + ": no tensor output");
    return keep[0].toTensor();
  }

  void lookup_row(uint32_t tok_id, float* dst) const {
    if (tok_id >= kEgVocab) throw std::runtime_error("embgemma: token id out of range");
    const __fp16* src = embed_table + static_cast<size_t>(tok_id) * kEgHidden;
    for (int j = 0; j < kEgHidden; ++j) dst[j] = static_cast<float>(src[j]) * kEgEmbedScale;
  }

  // Smallest batch-1 bucket that holds n tokens.
  const TextBucket& bucket_for(int n) const {
    for (const auto& b : buckets) {
      if (b.B == 1 && b.S >= n) return b;
    }
    throw std::runtime_error("embgemma: sequence of " + std::to_string(n) + " tokens exceeds 512");
  }

  const TextBucket* batch_bucket(int B, int n) const {
    for (const auto& b : buckets) {
      if (b.B == B && b.S >= n) return &b;
    }
    return nullptr;
  }

  // Runs [bk] on embeds [B,S,512] whose first n[i] rows per item are valid -> out[B*768].
  void run_text(const TextBucket& bk, std::vector<float>& embeds, const int* n, float* out,
                uint64_t& text_us) {
    std::vector<float> mask(static_cast<size_t>(bk.B) * bk.S, 0.f);
    for (int i = 0; i < bk.B; ++i) std::fill_n(mask.begin() + static_cast<size_t>(i) * bk.S, n[i], 1.f);
    auto et = make_tensor_ptr({bk.B, bk.S, kEgHidden}, embeds.data(), ScalarType::Float);
    auto mt = make_tensor_ptr({bk.B, bk.S}, mask.data(), ScalarType::Float);
    const auto& o = run(bk.file, bk.method, {EValue(*et), EValue(*mt)}, "embgemma text", text_us);
    if (o.numel() != static_cast<ssize_t>(bk.B) * kEgOutDim) {
      throw std::runtime_error("embgemma text: unexpected output size");
    }
    memcpy(out, o.const_data_ptr<float>(), static_cast<size_t>(bk.B) * kEgOutDim * sizeof(float));
  }

  // Writes [BOS, BOx, soft x n, EOx, EOS] at dst (n + 4 rows).
  void put_soft_tokens(const float* soft, int n, uint32_t bo, uint32_t eo, float* dst) const {
    lookup_row(kEgBos, dst);
    lookup_row(bo, dst + kEgHidden);
    memcpy(dst + 2 * kEgHidden, soft, static_cast<size_t>(n) * kEgHidden * sizeof(float));
    lookup_row(eo, dst + static_cast<size_t>(2 + n) * kEgHidden);
    lookup_row(kEgEos, dst + static_cast<size_t>(3 + n) * kEgHidden);
  }

  // [BOS, BOx, soft x n, EOx, EOS] -> smallest text bucket -> out[768].
  void embed_soft_tokens(const float* soft, int n, uint32_t bo, uint32_t eo, float* out,
                         uint64_t& text_us) {
    int total = n + 4;
    const TextBucket& bk = bucket_for(total);
    std::vector<float> embeds(static_cast<size_t>(bk.S) * kEgHidden, 0.f);
    put_soft_tokens(soft, n, bo, eo, embeds.data());
    run_text(bk, embeds, &total, out, text_us);
  }
};

std::unique_ptr<Engine> g_eng;

Engine& eng() {
  if (!g_eng) throw std::runtime_error("EmbGemmaNpu.init() not called");
  return *g_eng;
}

void open_engine(const std::string& dir) {
  auto e = std::make_unique<Engine>();
  e->dir = dir;
  std::string emb_path = dir + "/embed_tokens_262144x512_f16.bin";
  e->embed_bytes = kEgVocab * kEgHidden * sizeof(__fp16);
  e->embed_fd = ::open(emb_path.c_str(), O_RDONLY);
  if (e->embed_fd < 0) throw std::runtime_error("cannot open " + emb_path);
  struct stat st {};
  if (fstat(e->embed_fd, &st) != 0 || static_cast<size_t>(st.st_size) != e->embed_bytes) {
    throw std::runtime_error("embed_tokens_262144x512_f16.bin size mismatch");
  }
  void* p = mmap(nullptr, e->embed_bytes, PROT_READ, MAP_PRIVATE, e->embed_fd, 0);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + emb_path);
  e->embed_table = static_cast<const __fp16*>(p);

  uint64_t t0 = now_us();
  e->tok = std::make_unique<tokenizers::HFTokenizer>();
  if (e->tok->load(dir + "/tokenizer.json") != tokenizers::Error::Ok) {
    throw std::runtime_error("failed to load " + dir + "/tokenizer.json");
  }
  LOGI("loaded tokenizer in %.1f ms", (now_us() - t0) / 1000.0);

  auto fb = read_file(dir + "/gemma4_mel_filters_128x257.f32");
  if (fb.size() != embgemma_mel::kMels * embgemma_mel::kBins * sizeof(float)) {
    throw std::runtime_error("gemma4_mel_filters_128x257.f32: expected f32 [128,257]");
  }
  e->mel_filters.resize(fb.size() / sizeof(float));
  memcpy(e->mel_filters.data(), fb.data(), fb.size());

  if (access((dir + "/" + kTextMulti).c_str(), R_OK) == 0) {
    e->buckets = {{kTextMulti, "q32", 1, 32},     {kTextMulti, "a160", 1, 160},
                  {kTextMulti, "img288", 1, 288}, {kTextMulti, "s512", 1, 512},
                  {kTextMulti, "grid", 9, 64}};
  } else {
    e->buckets = {{"embgemma2_text_S128_qnn_fp16.pte", "forward", 1, 128},
                  {"embgemma2_text_S512_qnn_fp16.pte", "forward", 1, 512}};
  }
  LOGI("text buckets: %s", e->buckets[0].file.c_str());
  g_eng = std::move(e);
}

void set_perf_mode(int mode) {
  if (mode < 0) return;
  executorch::runtime::BackendOptions<2> opts;
  if (opts.set_option(QNN_RUNTIME_HTP_PERFORMANCE_MODE, mode) != Error::Ok ||
      executorch::runtime::set_option(QNN_BACKEND, opts.view()) != Error::Ok) {
    LOGE("failed to set htp performance mode %d", mode);
  } else {
    LOGI("htp performance mode %d", mode);
  }
}

void throw_java(JNIEnv* env, const char* msg) {
  env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), msg);
}

void put_times(JNIEnv* env, jlongArray times, std::initializer_list<jlong> v) {
  if (times && env->GetArrayLength(times) >= static_cast<jsize>(v.size())) {
    env->SetLongArrayRegion(times, 0, static_cast<jsize>(v.size()), v.begin());
  }
}

template <typename F>
auto guarded(JNIEnv* env, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const std::exception& ex) {
    LOGE("%s", ex.what());
    throw_java(env, ex.what());
  }
  return decltype(f())();
}

}  // namespace

extern "C" {

// Opens the CPU assets (tokenizer, embed table, mel filters) in [dir] and sets the HTP
// performance mode (-1 keeps the AOT default). Models load lazily, or eagerly via preload().
JNIEXPORT void JNICALL Java_com_znasif_xrpanosearch_npu_EmbGemmaNpu_nativeInit(
    JNIEnv* env, jclass, jstring jdir, jint perf_mode) {
  guarded(env, [&] {
    executorch::runtime::runtime_init();
    const char* d = env->GetStringUTFChars(jdir, nullptr);
    std::string dir(d);
    env->ReleaseStringUTFChars(jdir, d);
    set_perf_mode(perf_mode);
    open_engine(dir);
    return 0;
  });
}

// Loads one .pte and all its methods (file name relative to the init dir); returns microseconds.
JNIEXPORT jlong JNICALL Java_com_znasif_xrpanosearch_npu_EmbGemmaNpu_nativePreload(
    JNIEnv* env, jclass, jstring jfile) {
  return guarded(env, [&]() -> jlong {
    const char* f = env->GetStringUTFChars(jfile, nullptr);
    std::string file(f);
    env->ReleaseStringUTFChars(jfile, f);
    uint64_t t0 = now_us();
    eng().model(file, "");
    return static_cast<jlong>(now_us() - t0);
  });
}

// text (caller includes the task prefix) -> emb[768]. times: {textUs, nTokens}.
JNIEXPORT jfloatArray JNICALL Java_com_znasif_xrpanosearch_npu_EmbGemmaNpu_nativeText(
    JNIEnv* env, jclass, jstring jtext, jlongArray times) {
  return guarded(env, [&]() -> jfloatArray {
    Engine& e = eng();
    const char* t = env->GetStringUTFChars(jtext, nullptr);
    std::string text(t);
    env->ReleaseStringUTFChars(jtext, t);

    auto enc = e.tok->encode(text, 1, 1);
    if (!enc.ok()) throw std::runtime_error("embgemma: tokenizer encode failed");
    std::vector<uint64_t> ids = std::move(enc.get());
    if (ids.empty() || ids.front() != kEgBos) ids.insert(ids.begin(), kEgBos);
    if (ids.back() != kEgEos) ids.push_back(kEgEos);
    if (ids.size() > 512) {
      ids.resize(512);
      ids.back() = kEgEos;
    }
    int n = static_cast<int>(ids.size());
    const TextBucket& bk = e.bucket_for(n);
    std::vector<float> embeds(static_cast<size_t>(bk.S) * kEgHidden, 0.f);
    for (int i = 0; i < n; ++i) {
      e.lookup_row(static_cast<uint32_t>(ids[i]), embeds.data() + static_cast<size_t>(i) * kEgHidden);
    }
    uint64_t text_us = 0;
    float out[kEgOutDim];
    e.run_text(bk, embeds, &n, out, text_us);

    put_times(env, times, {static_cast<jlong>(text_us), n});
    jfloatArray res = env->NewFloatArray(kEgOutDim);
    env->SetFloatArrayRegion(res, 0, kEgOutDim, out);
    return res;
  });
}

// pixels f32 [1,3,672,960] RGB in [0,1] -> 1 (full) or 10 (full + 3x3 grid) x [768].
// times: {visionUs, textUs}.
JNIEXPORT jfloatArray JNICALL Java_com_znasif_xrpanosearch_npu_EmbGemmaNpu_nativeImageGrid(
    JNIEnv* env, jclass, jfloatArray jpixels, jboolean include_grid, jlongArray times) {
  return guarded(env, [&]() -> jfloatArray {
    Engine& e = eng();
    const size_t numel = static_cast<size_t>(3) * kVisH * kVisW;
    if (static_cast<size_t>(env->GetArrayLength(jpixels)) != numel) {
      throw std::runtime_error("embgemma vision: pixels must be f32 [1,3,672,960]");
    }
    uint64_t vis_us = 0, text_us = 0;
    std::vector<float> soft(static_cast<size_t>(kVisTokens) * kEgHidden);
    {
      std::vector<float> px(numel);
      env->GetFloatArrayRegion(jpixels, 0, static_cast<jsize>(numel), px.data());
      auto pt = make_tensor_ptr({1, 3, kVisH, kVisW}, px.data(), ScalarType::Float);
      const auto& feats = e.run("embgemma2_vision_landscape_qnn_fp16.pte", "forward", {EValue(*pt)},
                                "embgemma vision", vis_us);
      if (feats.numel() != static_cast<ssize_t>(soft.size())) {
        throw std::runtime_error("embgemma vision: unexpected output size");
      }
      memcpy(soft.data(), feats.const_data_ptr<float>(), soft.size() * sizeof(float));
    }

    int n_embs = include_grid ? 10 : 1;
    std::vector<float> embs(static_cast<size_t>(n_embs) * kEgOutDim);
    e.embed_soft_tokens(soft.data(), kVisTokens, kEgBoi, kEgEoi, embs.data(), text_us);

    if (include_grid) {
      // 3x3 overlapping windows of [6, 8] = 48 tokens over the [14, 20] soft grid, row-major
      // (Top-L..Bot-R), row starts {0, 4, 8}, col starts {0, 6, 12}.
      constexpr int GH = 14, GW = 20, sh = 6, sw = 8;
      const int r_starts[3] = {0, (GH - sh) / 2, GH - sh};
      const int c_starts[3] = {0, (GW - sw) / 2, GW - sw};
      constexpr int n = sh * sw;
      auto copy_window = [&](int r0, int c0, float* dst) {
        for (int dr = 0; dr < sh; ++dr) {
          memcpy(dst + static_cast<size_t>(dr * sw) * kEgHidden,
                 soft.data() + static_cast<size_t>((r0 + dr) * GW + c0) * kEgHidden,
                 static_cast<size_t>(sw) * kEgHidden * sizeof(float));
        }
      };
      if (const TextBucket* bk = e.batch_bucket(9, n + 4)) {
        // All 9 windows in one batched call: [9, S, 512].
        const size_t item = static_cast<size_t>(bk->S) * kEgHidden;
        std::vector<float> embeds(9 * item, 0.f), patch(static_cast<size_t>(n) * kEgHidden);
        int lens[9];
        int idx = 0;
        for (int r0 : r_starts) {
          for (int c0 : c_starts) {
            copy_window(r0, c0, patch.data());
            e.put_soft_tokens(patch.data(), n, kEgBoi, kEgEoi, embeds.data() + idx * item);
            lens[idx++] = n + 4;
          }
        }
        e.run_text(*bk, embeds, lens, embs.data() + kEgOutDim, text_us);
      } else {
        std::vector<float> patch(static_cast<size_t>(n) * kEgHidden);
        int idx = 1;
        for (int r0 : r_starts) {
          for (int c0 : c_starts) {
            copy_window(r0, c0, patch.data());
            e.embed_soft_tokens(patch.data(), n, kEgBoi, kEgEoi,
                                embs.data() + static_cast<size_t>(idx++) * kEgOutDim, text_us);
          }
        }
      }
    }
    LOGI("image (grid=%d): vision %.0f ms, text %.0f ms", include_grid, vis_us / 1000.0,
         text_us / 1000.0);

    put_times(env, times, {static_cast<jlong>(vis_us), static_cast<jlong>(text_us)});
    jfloatArray res = env->NewFloatArray(static_cast<jsize>(embs.size()));
    env->SetFloatArrayRegion(res, 0, static_cast<jsize>(embs.size()), embs.data());
    return res;
  });
}

// 16 kHz mono s16 PCM (<= 5 s used) -> emb[768]. times: {melUs, audioUs, textUs, nSoftTokens}.
JNIEXPORT jfloatArray JNICALL Java_com_znasif_xrpanosearch_npu_EmbGemmaNpu_nativeAudio(
    JNIEnv* env, jclass, jshortArray jpcm, jlongArray times) {
  return guarded(env, [&]() -> jfloatArray {
    Engine& e = eng();
    std::vector<int16_t> pcm(static_cast<size_t>(env->GetArrayLength(jpcm)));
    env->GetShortArrayRegion(jpcm, 0, static_cast<jsize>(pcm.size()), pcm.data());

    uint64_t t0 = now_us();
    auto mel = embgemma_mel::compute(pcm.data(), static_cast<int>(pcm.size()), kAudT,
                                     e.mel_filters.data());
    uint64_t mel_us = now_us() - t0, aud_us = 0, text_us = 0;

    int nt = std::clamp(mel.n_tokens, 1, kAudN);
    std::vector<float> soft(static_cast<size_t>(kAudN) * kEgHidden);
    {
      auto mt = make_tensor_ptr({1, kAudT, embgemma_mel::kMels}, mel.mel.data(), ScalarType::Float);
      auto mkt = make_tensor_ptr({1, kAudT}, mel.mask.data(), ScalarType::Float);
      const auto& feats = e.run("embgemma2_audio_5s_qnn_fp16.pte", "forward", {EValue(*mt), EValue(*mkt)},
                                "embgemma audio", aud_us);
      if (feats.numel() != static_cast<ssize_t>(soft.size())) {
        throw std::runtime_error("embgemma audio: unexpected output size");
      }
      memcpy(soft.data(), feats.const_data_ptr<float>(), soft.size() * sizeof(float));
    }
    float out[kEgOutDim];
    e.embed_soft_tokens(soft.data(), nt, kEgBoa, kEgEoa, out, text_us);
    LOGI("audio (%.2f s -> %d soft tok): mel %.1f ms, audio %.0f ms, text %.0f ms",
         pcm.size() / 16000.0, nt, mel_us / 1000.0, aud_us / 1000.0, text_us / 1000.0);

    put_times(env, times, {static_cast<jlong>(mel_us), static_cast<jlong>(aud_us),
                           static_cast<jlong>(text_us), nt});
    jfloatArray res = env->NewFloatArray(kEgOutDim);
    env->SetFloatArrayRegion(res, 0, kEgOutDim, out);
    return res;
  });
}

}  // extern "C"
