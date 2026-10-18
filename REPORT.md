# EmbeddingGemma 2 & XR Panoramic Spatial Search — Progress & Next Steps Report

**Target Device:** Samsung Galaxy XR (`R3GYB01AKWW`, SoC `SXR2230P`, Hexagon V69 `v69`, Firmware `I610UEU1AYIA`, Android 14 / SDK 34)  
**Model:** `google/embeddinggemma-2` (Gemma-4 multimodal embedding model: 24L text backbone + 16L Gemma-4 vision tower + 12L conformer audio tower → shared L2-normalized **768-d** embedding space)

---

## 1. What Has Been Completed & Verified on Device

### Phase 1: EmbeddingGemma 2 Hexagon V69 NPU Exports (`100% Complete`)
All three modalities of `google/embeddinggemma-2` were decomposed, exported via ExecuTorch + QNN 2.46 (`QnnHtp.so` v69 FP16), and verified on `R3GYB01AKWW` with **zero CPU fallback ops** (`delegated_calls = 1`, `cpu_ops = 0`):

| Modality | Reports & Scripts | `.pte` Artifacts (`/data/local/tmp/npu/embgemma2/`) | Accuracy vs PyTorch Golden | On-Device NPU Latency |
|---|---|---|---|---|
| **Text + Shared Backbone** | `/home/znasif/local/hackathon/embgemma2/text/REPORT.md` | `embgemma2_text_S128_qnn_fp16.pte` (284 MB)<br>`embgemma2_text_S512_qnn_fp16.pte` (297 MB)<br>`embed_tokens_262144x512_f16.bin` (256 MB) | `min_cos = 0.999997` | **49.6 ms** (`S128`)<br>**250.7 ms** (`S512`, qnn_executor_runner; ~204 ms measured inside npu_server) |
| **Vision Tower + 3×3 Grid** | `/home/znasif/local/hackathon/embgemma2/vision/REPORT.md` | `embgemma2_vision_landscape_qnn_fp16.pte` (`672×960 → [1,280,512]`, 324 MB)<br>`embgemma2_vision_portrait_qnn_fp16.pte` (`960×672 → [1,280,512]`, 324 MB) | final-embedding `min_cos = 0.999984` vs fp32 (full)<br>3×3 grid: `cos = 1.000000` only vs its own host reimplementation — **retrieval quality of pooled sub-regions not yet validated** | **773 ms** vision + **206 ms** full text + **9 × 54 ms** (`487 ms`) 3×3 grid = **1.47 s** total for **10 embeddings** |
| **Audio Tower** | `/home/znasif/local/hackathon/embgemma2/audio/REPORT.md` | `embgemma2_audio_5s_qnn_fp16.pte` (`[1,500,128] → [1,125,512]`, 604 MB)<br>`embgemma2_audio_11s_qnn_fp16.pte` (`[1,1120,128] → [1,280,512]`, 609 MB) | final-embedding `min_cos = 0.999997` (per-token ≥ 0.998) | **67.3 ms** (`5s` audio) + **49.5 ms** (`S128` text) = **117 ms** NPU |

---

### Phase 2A: Native `npu_server` Daemon Ops (`8`, `9`, `10`) (`100% Complete`)
Extended the persistent C++ Unix-socket NPU daemon (`/home/znasif/local/hackathon/npu_server/npu_server.cpp`) so all three EmbeddingGemma 2 pipelines run natively in C++ with zero Python dependency:
- **`embgemma_mel.h`**: Standalone header-only C++ port of Gemma-4's `Gemma4AudioFeatureExtractor` (16 kHz s16le PCM → 160-zero pre-pad → semicausal 320-sample frames / 160 hop → periodic `float64` Hann window → 512-pt radix-2 FFT magnitude → `[128, 257]` HTK triangular mel filterbank → `log(mel + 0.001)`).
- **`EMBGEMMA_TEXT = 8`**: C++ `HFTokenizer` (`tokenizer.json`) + `mmap`-ed `embed_tokens_262144x512_f16.bin` (`* 22.627417f`) → `text_S128` or `text_S512` → `[768]` float32.
- **`EMBGEMMA_IMAGE_GRID = 9`**: Accepts `[1, 3, H, W]` float32 (`orient=0` `672×960` or `orient=1` `960×672`) + `flags` (`bit0 = include 3×3 grid`), runs `vision_{landscape,portrait}` once, runs `text_S512` on all 280 tokens, and runs `text_S128` 9 times on the `6×8=48` sub-grids (`rows {0..5, 4..9, 8..13} × cols {0..7, 6..13, 12..19}`), returning `1` or `10 × [768]` float32.
- **`EMBGEMMA_AUDIO = 10`**: Accepts 16 kHz s16le PCM (or precomputed `[1, B, 128]` mel), runs `embgemma_mel::compute` (`~9 ms` CPU), `audio_5s` or `audio_11s`, splices the valid soft tokens as `[BOS=2, BOA=256000, soft×n, EOA=258883, EOS=1]` (`eg_embed_soft_tokens`; images use BOI=255999 / EOI=258882), and runs `text_S128` / `text_S512` → `[768]` float32.
- Verified all 3 ops over `adb forward` via `npu_server/test_embgemma.py` (`min_cos = 0.999992`).

---

### Phase 2B: `npuclient` Android AAR & Shizuku Bridge Verification (`100% Complete`)
Updated `com.google.xr.embardiment.npu` (`INpuService.aidl`, `NpuUserService.java`, `NpuTensor.kt`, `NpuClient.kt`, `SERVICE_VERSION = 6`) and verified on `R3GYB01AKWW` via `EmbGemmaGateActivity`:
```text
EmbGemmaGate: status: READY: npu_server 3 qairt 2.46
EmbGemmaGate: TEXT nTok=15 cold=2195 ms warm=52 ms (npu=49 ms) cos=1.000000
EmbGemmaGate: IMAGE_GRID gridSize=9 vis=773 ms text=693 ms wall=2124 ms cosFull=1.000000 minGridCos=1.000000
EmbGemmaGate: AUDIO nSoft=41 mel=9 ms aud=67 ms text=49 ms wall=506 ms cos=1.000000
EmbGemmaGate: ALL_DONE PASS
```

---

### Phase 2C: Standalone Jetpack XR Project Scaffold (`In Progress — ~25% Complete`)
Created `/home/znasif/local/hackathon/xr-panosearch/` with Gradle configuration, `npuclient-release.aar`, spatial audio cues, and the `sweep_golden` room capture dataset.

---

## 2. What Needs To Be Done Next

1. **Step 1: Jetpack XR Runtime + Camera2 `"0"` Probe on Samsung Galaxy XR**
   - Create `AndroidManifest.xml` and `GalaxyXrCamera.kt` (explicitly targeting Galaxy XR world Camera2 ID `"0"` at `1920×1440` with graceful off-head `CAMERA_DISABLED` handling) and verify `androidx.xr.compose` (`Subspace`, `SpatialPanel`, `LocalSession`) launches cleanly on `R3GYB01AKWW`.
2. **Step 2: 360° Panoramic Ring Capture & Background V69 NPU Grid Indexer**
   - Build the 12-panel 360° ring (`30°` yaw steps at `1.0 m` radius) supporting both live head-pose Camera `"0"` capture and one-tap `sweep_golden` room replay, indexing each panel on a background worker thread into `1 + 9 = 10` × 768-d vectors via `NpuClient.embgemmaImageGrid`.
3. **Step 3: Gesture (Pinch / Tap) + Voice Query Pipeline**
   - Implement 16 kHz mono mic capture (`AudioRecord` with endpointing) triggered by pinch/tap in XR Full Space, supporting both direct audio embedding (`embgemmaAudio`) and Whisper transcript + text embedding (`whisperTranscribe` + `embgemmaText`), plus quick preset query chips.
4. **Step 4: Multi-Element 3D World-Locked Spatial Anchors**
   - Score all `12 × 9 = 108` sub-regions against the query vector, apply 2D neighborhood NMS, and place world-locked 3D `SpatialPanel` anchor cards at the exact 3D `(x, y, z, -yaw)` coordinates of every scene sub-region exceeding the similarity threshold.
5. **Step 5: On-Device Build, Install & End-to-End Verification on `R3GYB01AKWW`**

---

## 3. Update 2026-10-18 (Claude Code session)

Report corrections: the earlier figures (`.pte` sizes, audio 11 s shape, audio token splice) were wrong and are fixed in section 1. Grid sub-region retrieval quality is **not validated yet**; it's being evaluated on host in `embgemma2/grid_eval/`.

App (Steps 1-4 code done, builds, installed on R3GYB01AKWW; Shizuku approved for `com.znasif.xrpanosearch`):
- Jetpack XR: compose `1.0.0-beta01`, scenecore/runtime/arcore `1.0.0-rc02`; Kotlin 2.1.21. Build: `JAVA_HOME=~/anaconda3/envs/executorch/opt/jdk-21.0.5+11 ~/.gradle/wrapper/dists/gradle-8.11.1-bin/*/gradle-8.11.1/bin/gradle assembleDebug`.
- `MainActivity` (Full Space): head pose from `ArDevice` (perception → activity space), live 12-slot ring capture from Camera2 `"0"` (copied `cam/WorldCamera.kt`, `cam/YuvFrame.kt` from the worldcam AAR) when the head is steady within 8° of a slot centre; "Replay recorded sweep" indexes `sweep_golden` (covers yaw −74…+18°, so slots 0°, +30°, −30°, −60°). Query by chip (text), "Speak (audio embedding)" or "Speak (Whisper)". Ring cells and result cards are SceneCore `PanelEntity`s with explicit activity-space poses (`WorldPanels.kt`); cards sit 1.8 m along the region-centre ray (nominal camera f = 0.604 × width, camera assumed aligned with head forward). Compose's Subspace placement APIs needed for world-locking are internal in beta01, hence PanelEntity.
- `ImagePrep.kt`: PIL-style antialiased bicubic; matches PIL within 1 level on `frame_0034` (checked on device dump).
- Off-head gate `ReplayGateActivity` (2D): NPU READY, 3 slots indexed (vision ~1.3 s, grid text ~1.37 s, wall ~2.7 s per slot), text queries ~100 ms warm. Region scores were flat (0.63-0.71 for every query), pending the grid evaluation.
- Not yet verified in the headset (needs wearing): Full Space launch, panel placement/orientation, live capture, mic.

### Update 2 (2026-10-18, verified on head)
- Ring rebuilt on Project_Aura geometry (`Ring.kt`): 12 panels tangent at R = 1.0 m, 0.536 × 0.728 m, each showing only its 30° × 40° camera slice, so the ring reads as a continuous world-aligned panorama. The faced panel is a live window (slice re-projected for head yaw/pitch); a 1.5 s dwell with rising ticks captures it (chime). The user confirmed the alignment and the full flow work.
- All world panels are SceneCore `PanelEntity`s parented to `activitySpace` (the parent-less create() overloads are never drawn). Control panel = ComposeView in a PanelEntity, 0.6 m ahead / 0.42 m low / tilted 40° up, re-centred after 45° of head turn.
- Search ranks the best grid cell only (whole-frame embedding excluded), per `embgemma2/grid_eval/REPORT.md` (cell top-1 0.87 within-frame, 8/10 cross-frame; crop+re-embed is no better at 7× the cost).
- Live session: 7 slots indexed at ~2.7 s each (prep ~470 ms, vision ~1.3 s, grid text ~1.4 s); text queries ~110 ms.

### Update 3 (2026-10-18): optimizations + cleanup
- NPU time per slot depends on headset state, not on the app: off-head (asleep) the app path measures vision 776 ms + grid text 703 ms (= `embgemma_bench` from adb shell, = `qnn_executor_runner` 776 ms); while worn it was ~1.3 s + ~1.4 s. Most likely the headset's own tracking shares the Hexagon DSP while worn; not confirmed and not fixable from the app.
- `ImagePrep` resize parallelized over 6 threads: 340-460 ms -> 175-280 ms, output unchanged (still within 1 level of PIL).
- Live-panel and capture-frame YUV->RGB conversions moved off the UI thread (`pano-live` executor / NPU worker), so panels no longer stutter during live capture or at the moment of capture.
- Result cards: up to 3 hits within 0.01 of the best (was 4 within 0.03); cell scores in one room sit within ~0.03 of each other, so the wider margin mostly added wrong cards.
- Debug PROBE panel removed (the ring rewrite dropped it). `ReplayGateActivity` stays as the off-head test.
