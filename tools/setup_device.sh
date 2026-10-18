#!/usr/bin/env bash
# One-time (and after-reboot) device setup for PanoSearch on Samsung Galaxy XR.
#
#   1. downloads the V69 models + npu_server from Hugging Face ($HF_REPO, or LOCAL_DIR),
#   2. downloads the Qualcomm QNN 2.46.0 runtime from Maven Central,
#   3. pushes everything to /data/local/tmp/npu/ (skips files already there with the same size),
#   4. installs Shizuku 13.6.0 if missing and starts its server (needed after every reboot).
#
# Usage:  tools/setup_device.sh            (full setup)
#         tools/setup_device.sh --shizuku  (after a reboot: only restart the Shizuku server)
# Env:    SERIAL=<adb serial> (if more than one device), HF_REPO=<user>/<repo>,
#         LOCAL_DIR=<folder with npu/> (use local files instead of Hugging Face).
set -euo pipefail
HF_REPO=${HF_REPO:-Znasif/embeddinggemma-2-qnn-v69}
QNN_VERSION=2.46.0
SHIZUKU_URL=https://github.com/RikkaApps/Shizuku/releases/download/v13.6.0/shizuku-v13.6.0.r1086.2650830c-release.apk
DST=/data/local/tmp/npu
CACHE=${CACHE:-$HOME/.cache/panosearch}
ADB=(adb); [[ -n "${SERIAL:-}" ]] && ADB=(adb -s "$SERIAL")

# adb.exe called from WSL needs Windows paths for local files.
winpath=0
if command -v wslpath >/dev/null && grep -qs 'adb.exe' "$(command -v adb)"; then winpath=1; fi
lp() { if [[ $winpath == 1 ]]; then wslpath -w "$1"; else echo "$1"; fi; }

die() { echo "ERROR: $*" >&2; exit 1; }

start_shizuku() {
  if "${ADB[@]}" shell 'ps -A -Z' | grep -q 'u:r:shell:s0.*shizuku_server'; then
    echo "Shizuku server already running"; return
  fi
  "${ADB[@]}" shell '$(dirname $(pm path moe.shizuku.privileged.api | cut -d: -f2))/lib/arm64/libshizuku.so' >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    if "${ADB[@]}" shell 'ps -A -Z' | grep 'shizuku_server' | grep -q 'u:r:shell:s0'; then
      echo "Shizuku server started (u:r:shell:s0)"; return
    fi
    sleep 0.5
  done
  die "Shizuku server did not start; open the Shizuku app once on the headset and retry"
}

"${ADB[@]}" get-state >/dev/null 2>&1 || die "no adb device (enable USB debugging, accept the prompt in the headset, set SERIAL if several)"
model=$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r')
soc=$("${ADB[@]}" shell getprop ro.soc.model | tr -d '\r')
echo "device: $model ($soc)"
[[ "$soc" == "SXR2230P" ]] || echo "WARNING: models are compiled for SXR2230P (Hexagon V69); $soc may not run them"

if [[ "${1:-}" == "--shizuku" ]]; then start_shizuku; exit 0; fi

# 1. Models + server.
mkdir -p "$CACHE"
if [[ -n "${LOCAL_DIR:-}" ]]; then
  SRC=$LOCAL_DIR
else
  command -v hf >/dev/null || die "install the Hugging Face CLI: pip install -U huggingface_hub"
  SRC=$CACHE/hf
  hf download "$HF_REPO" --local-dir "$SRC" >/dev/null
fi
[[ -f "$SRC/npu/npu_server" ]] || die "$SRC/npu/npu_server missing"

# 2. QNN runtime (Qualcomm license, see NOTICE.txt / LICENSE.pdf in the AAR).
QNN=$CACHE/qnn-runtime-$QNN_VERSION
if [[ ! -f "$QNN/libQnnHtpV69Skel.so" ]]; then
  mkdir -p "$QNN"
  curl -fsSL -o "$QNN/qnn-runtime.aar" \
    "https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/$QNN_VERSION/qnn-runtime-$QNN_VERSION.aar"
  for f in libQnnHtp.so libQnnHtpPrepare.so libQnnHtpV69Skel.so libQnnHtpV69Stub.so libQnnSystem.so; do
    unzip -o -j -q "$QNN/qnn-runtime.aar" "jni/arm64-v8a/$f" -d "$QNN"
  done
fi

# 3. Push (path relative to $DST <- local file).
"${ADB[@]}" shell mkdir -p $DST/embgemma2
declare -A DEV
while read -r name size; do DEV[$name]=$size; done < <("${ADB[@]}" shell \
  "cd $DST && for f in * embgemma2/*; do [ -f \"\$f\" ] && echo \"\$f \$(stat -c %s \"\$f\")\"; done" | tr -d '\r')
pushed=0
push() { # <local file> <name relative to $DST>
  local size; size=$(stat -c %s "$1" 2>/dev/null || stat -f %z "$1")
  if [[ "${DEV[$2]:-}" == "$size" ]]; then echo "ok    $2"; return; fi
  echo "push  $2 ($((size / 1048576)) MB)"
  "${ADB[@]}" push "$(lp "$1")" "$DST/$2" >/dev/null 2>&1
  pushed=1
}
for f in "$SRC"/npu/*; do [[ -f "$f" ]] && push "$f" "$(basename "$f")"; done
for f in "$SRC"/npu/embgemma2/*; do push "$f" "embgemma2/$(basename "$f")"; done
for f in "$QNN"/*.so; do push "$f" "$(basename "$f")"; done
"${ADB[@]}" shell chmod 755 $DST/npu_server
# A running server keeps old files mapped; it restarts on the next app call.
if [[ $pushed == 1 ]]; then "${ADB[@]}" shell "pkill -f $DST/npu_server" >/dev/null 2>&1 || true; fi

# 4. Shizuku.
if ! "${ADB[@]}" shell pm path moe.shizuku.privileged.api >/dev/null 2>&1; then
  curl -fsSL -o "$CACHE/shizuku.apk" "$SHIZUKU_URL"
  "${ADB[@]}" install "$(lp "$CACHE/shizuku.apk")" >/dev/null
  echo "installed Shizuku 13.6.0"
fi
start_shizuku
echo "device ready: $DST"
