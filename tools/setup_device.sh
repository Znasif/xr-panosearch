#!/usr/bin/env bash
# One-time device setup for PanoSearch on Project Aura (SXR2330P, Hexagon V79).
#
# Pushes the V79 EmbeddingGemma 2 models and CPU assets into the app's own external files
# directory, /sdcard/Android/data/com.znasif.xrpanosearch/files/embgemma2/ (no root, no shell
# daemon: the app runs the models in-process). Files already there with the same size are skipped.
# The QNN runtime ships inside the APK (com.qualcomm.qti:qnn-runtime).
#
# Usage:  LOCAL_DIR=<folder with the files below> tools/setup_device.sh [--gate]
#         --gate also pushes LOCAL_DIR/gate/ (host fp32 references for ReplayGateActivity golden mode).
# Env:    SERIAL=<adb serial> (if more than one device).
# Files:  embgemma2_text_S128_qnn_fp16.pte  embgemma2_text_S512_qnn_fp16.pte
#         embgemma2_vision_landscape_qnn_fp16.pte  embgemma2_audio_5s_qnn_fp16.pte
#         tokenizer.json  embed_tokens_262144x512_f16.bin  gemma4_mel_filters_128x257.f32
set -euo pipefail
PKG=com.znasif.xrpanosearch
DST=/sdcard/Android/data/$PKG/files/embgemma2
FILES=(embgemma2_text_S128_qnn_fp16.pte embgemma2_text_S512_qnn_fp16.pte
  embgemma2_vision_landscape_qnn_fp16.pte embgemma2_audio_5s_qnn_fp16.pte
  tokenizer.json embed_tokens_262144x512_f16.bin gemma4_mel_filters_128x257.f32)
ADB=(adb); [[ -n "${SERIAL:-}" ]] && ADB=(adb -s "$SERIAL")

# adb.exe called from WSL needs Windows paths for local files.
winpath=0
if command -v wslpath >/dev/null && grep -qs 'adb.exe' "$(command -v adb)"; then winpath=1; fi
lp() { if [[ $winpath == 1 ]]; then wslpath -w "$1"; else echo "$1"; fi; }

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${LOCAL_DIR:-}" ]] || die "set LOCAL_DIR to the folder holding: ${FILES[*]}"
for f in "${FILES[@]}"; do [[ -f "$LOCAL_DIR/$f" ]] || die "$LOCAL_DIR/$f missing"; done

"${ADB[@]}" get-state >/dev/null 2>&1 || die "no adb device (set SERIAL if several)"
soc=$("${ADB[@]}" shell getprop ro.soc.model | tr -d '\r')
echo "device: $soc"
[[ "$soc" == "SXR2330P" ]] || echo "WARNING: models are compiled for SXR2330P (Hexagon V79); $soc may not run them"

"${ADB[@]}" shell mkdir -p $DST
declare -A DEV
while read -r name size; do DEV[$name]=$size; done < <("${ADB[@]}" shell \
  "cd $DST && for f in *; do [ -f \"\$f\" ] && echo \"\$f \$(stat -c %s \"\$f\")\"; done" | tr -d '\r')
for f in "${FILES[@]}"; do
  size=$(stat -c %s "$LOCAL_DIR/$f" 2>/dev/null || stat -f %z "$LOCAL_DIR/$f")
  if [[ "${DEV[$f]:-}" == "$size" ]]; then echo "ok    $f"; continue; fi
  echo "push  $f ($((size / 1048576)) MB)"
  "${ADB[@]}" push "$(lp "$LOCAL_DIR/$f")" "$DST/$f" >/dev/null
done
if [[ "${1:-}" == "--gate" ]]; then
  "${ADB[@]}" shell mkdir -p $DST/gate
  for f in "$LOCAL_DIR"/gate/*; do "${ADB[@]}" push "$(lp "$f")" "$DST/gate/" >/dev/null; echo "push  gate/$(basename "$f")"; done
fi
echo "device ready: $DST (force-stop the app if it already had models loaded)"
