# PanoSearch

Search your surroundings by text or voice on a **Samsung Galaxy XR**, with every model running on the headset's **Hexagon NPU**. Nothing goes to the cloud.

1. You turn your head around the room. A ring of 12 panels at eye level fills in, one 30° slice at a time, into a panorama that lines up with the real world.
2. Each captured slice is embedded on the NPU with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): one embedding for the full frame, plus one for each of 9 sub-regions.
3. You ask for something ("phone", "monitor", …) with a button, by voice (embedded directly from audio), or by voice transcribed with Whisper.
4. Cards appear in the room in front of the best-matching regions.

<a href="https://youtube.com/shorts/uARrX3xB77o"><img src="https://img.youtube.com/vi/uARrX3xB77o/maxresdefault.jpg" alt="PanoSearch demo on Galaxy XR (YouTube)" width="480"></a>

▶ [Watch the demo on YouTube](https://youtube.com/shorts/uARrX3xB77o)

## What you need

| | |
|---|---|
| Headset | Samsung Galaxy XR (SM-I610, SoC SXR2230P / Hexagon V69). Firmware **I610UEU1AYIA**. **Don't install system updates:** on I610UEU2AZF3 the NPU is blocked even from adb shell. Keep Wi-Fi off if you need to avoid the update. |
| On the headset | Developer options and USB debugging enabled; the USB-debugging prompt accepted. |
| Computer | `adb` (Android platform-tools), JDK 17 or newer, Android SDK with platform 36, `curl`, `unzip`, and the Hugging Face CLI (`pip install -U huggingface_hub`). Linux, macOS, or WSL with Windows `adb.exe` all work. |

## Setup (once)

```bash
git clone https://github.com/Znasif/xr-panosearch.git
cd xr-panosearch

# 1. Put the models, the NPU server and the Qualcomm runtime on the headset, and start Shizuku.
#    Downloads ~2.8 GB from Hugging Face (Znasif/embeddinggemma-2-qnn-v69) and the QNN 2.46.0
#    runtime from Maven Central, pushes them to /data/local/tmp/npu/, and installs Shizuku 13.6.0.
#    Re-running it only pushes what changed.
tools/setup_device.sh
#    (several adb devices? prefix SERIAL=<serial>; models already local? LOCAL_DIR=<folder containing npu/>)

# 2. Build and install the app.
export ANDROID_HOME=/path/to/android-sdk   # or create local.properties with sdk.dir=...
./gradlew installDebug
```

3. Put the headset on and open **PanoSearch** from the app launcher.
   - Allow the **Camera**, **Microphone**, and **head tracking / scene understanding** permissions.
   - A **Shizuku** dialog asks whether PanoSearch may use it. Choose **Allow** (once per install). This is what lets the app reach the NPU.

**After every headset reboot**, restart the Shizuku server before using the app:

```bash
tools/setup_device.sh --shizuku
```

## Using it

The control panel opens low and in front of you, tilted up. It moves back in front of you if you turn more than 45° away from it.

1. **Capture room (live):** the panel you face shows the camera.
   - Hold still about 1.5 s while the blue bar fills and the ticks speed up.
   - A chime plays and the panel freezes into its slice of the panorama.
   - Turn to the next panel. Indexing runs in the background, about 2.7 s per panel while worn.
   - Empty panels have a faint outline; capture as many as you like.
2. **Replay sweep:** fills the ring from a recorded sweep bundled in the app (`assets/sweep_golden`, slots 0°, ±30°, −60°). Useful for a demo without walking around.
3. **Find:** tap a chip (laptop, monitor, phone, …). Up to 3 cards appear in front of the matching regions.
4. **Speak (audio):** say what you're looking for (up to 5 s). The audio itself is embedded and compared, with no transcript.
5. **Speak (Whisper):** same, but Whisper transcribes it first and the text is embedded.
6. **Clear:** removes the cards.

## Check it without wearing the headset

Immersive apps pause when the headset is off. A 2D test activity runs the same NPU path on the recorded sweep:

```bash
adb logcat -c
adb shell am start -n com.znasif.xrpanosearch/.ReplayGateActivity
adb logcat -s PanoSearch:I     # ends with GATE_DONE PASS
```

Expected output (headset off):

```
NPU: READY
INDEX slot=0 ... vision=772ms text=697ms ...
Q "a laptop" ...
GATE_DONE PASS
```

## Troubleshooting

| Control panel / log says | Do this |
|---|---|
| `NPU: SHIZUKU_NOT_RUNNING` | `tools/setup_device.sh --shizuku` (needed after every reboot). |
| `NPU: SHIZUKU_NOT_APPROVED` / `SHIZUKU_DENIED` | Put the headset on and allow PanoSearch in the Shizuku dialog, or in the Shizuku app under *Authorized applications*. |
| `NPU: SERVICE_ERROR: ...` | Re-run `tools/setup_device.sh` (a file is missing or stale on the device), then reopen the app. |
| Panel says `too dark` / `look level at this panel` | More light; keep your head within about 35° of level while capturing. |
| `camera: DISABLED` / no live image | The camera only works while the headset is worn and the app is in front. |
| Wrong or extra cards | Scores for different regions of one room are close (within ~0.03), so try a more specific word. Objects cut off at a frame edge (a laptop seen from above) are the usual miss. |

## How it works

```
PanoSearch app (Jetpack XR, Kotlin)
  ring + cards: SceneCore PanelEntity in activity space
  camera: Camera2 id "0" (1920x1440 YUV)      head pose: ARCore for Jetpack XR ArDevice
  ImagePrep: antialiased bicubic -> [1,3,672,960]
      |  npuclient AAR (app/libs): Shizuku UserService + SharedMemory tensors
      v
NpuUserService  (runs inside shizuku_server as the shell user, the only app-reachable
      |          context with NPU access on this firmware)
      v
/data/local/tmp/npu/npu_server  (ExecuTorch + QNN 2.46, models loaded once)
   op 8  text   : tokenizer + embedding table (CPU) -> text .pte -> 768-d
   op 9  image  : vision .pte -> 280 soft tokens -> full-frame embedding (S512)
                  + 9 overlapping 6x8-token windows -> 9 region embeddings (S128)
   op 10 audio  : log-mel (CPU) -> audio .pte -> soft tokens -> text .pte -> 768-d
   op 6  whisper: base.en encoder/decoder
```

- **Ring geometry** (`Ring.kt`, after Project_Aura): 12 flat panels tangent to a 1.0 m circle. Each is `2·R·tan(15°)` = 0.536 m wide, so neighbours meet edge to edge, and shows only the camera pixels covering its own 30° × 40°. The camera model is nominal (f = 0.604 × image width, about 79° horizontal FOV), because Android XR exposes no calibrated intrinsics.
- **Search** (`PanoIndex.kt`): cosine between the query and every region embedding. Results are ranked by the best region, not by the whole frame, then de-duplicated by overlap and by direction. On a labeled sweep the best region contained the object 87% of the time (chance: 28%). That ties with cropping each region and re-embedding it, which costs 7× more.
- **Cards** sit 0.85 m along the camera ray through the region's centre, from where your head was at capture time.

| Step (Galaxy XR) | Headset off | Worn |
|---|---|---|
| Image resize (CPU, 6 threads) | 175–280 ms | similar |
| Vision tower (NPU) | 772 ms | ~1.3 s |
| Full frame + 9 regions, text model (NPU) | 700 ms | ~1.4 s |
| Text query | ~50 ms (NPU), ~110 ms round trip | similar |
| Spoken query, 5 s (mel + audio + text) | ~130 ms | |

## Repository layout

```
app/src/main/java/com/znasif/xrpanosearch/
  MainActivity.kt        capture loop, live window, queries, control panel
  Ring.kt                panorama ring geometry and per-panel camera crop
  WorldPanels.kt         ring panels, control panel, result cards (SceneCore)
  Engine.kt              NPU calls (index, text/audio/Whisper queries), replay frames
  PanoIndex.kt           regions, head frames, search
  ImagePrep.kt           PIL-style antialiased bicubic resize to the vision input
  MicRecorder.kt         16 kHz capture with energy endpointing
  ReplayGateActivity.kt  2D off-head test
  cam/                   Camera2 world camera (from the Embardiment worldcam module)
app/libs/npuclient-release.aar   NPU client (Shizuku + SharedMemory)
app/src/main/assets/   recorded sweep (sweep_golden) and capture sounds
tools/setup_device.sh  device setup
REPORT.md              development log and measurements
```

## Rebuilding the models or the server (maintainers)

You don't need this to use the app. The Hugging Face repo has everything. The sources are **not published yet**; they're on the development machine:

- `embgemma2/{text,vision,audio}/`: export scripts (`scripts/export_*_fp16.py`), host and device gates, and a `REPORT.md` per part with exact commands. They use the ExecuTorch + QAIRT 2.46 conda env, plus transformers 5.20.1 from `embgemma2/pydeps`.
- `npu_server/`: C++ server (`build.sh`, `PROTOCOL.md`).
- `npuclient`: in the Embardiment repo, `tools/AAR-Tooling/npuclient`.

Peak host RAM per export: text 5.7 GB, vision 7.9 GB, audio 9.1 GB.

## Licenses and credits

- App code: see `LICENSE` (to be added by the repo owner).
- Model files: EmbeddingGemma 2 derivatives under Apache-2.0 (Google); Whisper base.en derivatives under MIT (OpenAI).
- Runtimes: ExecuTorch under BSD-3-Clause. The Qualcomm QNN runtime is downloaded from Maven Central under Qualcomm's license and isn't redistributed here.
- [Shizuku](https://github.com/RikkaApps/Shizuku) under Apache-2.0.
- Ring geometry adapted from Project_Aura.

**Privacy:** live camera frames stay on the headset and aren't saved. The bundled `sweep_golden` frames are photos of one room.
