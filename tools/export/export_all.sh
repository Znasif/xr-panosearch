#!/usr/bin/env bash
# Re-export the EmbeddingGemma 2 .pte set used by PanoSearch for Project Aura (SXR2330P, HTP V79).
set -uo pipefail
H=/home/znasif/local/hackathon/embgemma2
source $H/text/scripts/env.sh
O=$H/v79; L=$O/logs
run() { local name=$1 dir=$2; shift 2
  ( cd $H/$dir/scripts && flock $H/.export.lock /usr/bin/time -v python "$@" ) > $L/$name.log 2>&1
  echo "$name exit=$? $(grep -h 'DELEGATION\|^OK' $L/$name.log | tr '\n' ' ')"; }
run text_S128 text export_text_fp16.py --S 128 --soc SXR2330P --out $O/embgemma2_text_S128_qnn_fp16.pte
run text_S512 text export_text_fp16.py --S 512 --soc SXR2330P --out $O/embgemma2_text_S512_qnn_fp16.pte
run vision_landscape vision export_vision_fp16.py --orient landscape --soc SXR2330P --out $O/embgemma2_vision_landscape_qnn_fp16.pte
run audio_5s audio export_audio_fp16.py --bucket 5s --soc SXR2330P --out $O/embgemma2_audio_5s_qnn_fp16.pte
echo ALL_EXPORTS_DONE
