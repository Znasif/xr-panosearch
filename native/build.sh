#!/usr/bin/env bash
# Builds libembgemma_npu.so (JNI) for arm64 Android against the ExecuTorch build-android/ outputs
# (QNN backend, Module/Tensor extensions, HF tokenizer) and copies it, together with
# libqnn_executorch_backend.so, into app/src/main/jniLibs/arm64-v8a/.
# The QNN runtime itself (libQnnHtp.so, V79 stub/skel, libQnnSystem.so) comes from the
# com.qualcomm.qti:qnn-runtime Maven dependency.
set -euo pipefail
cd "$(dirname "$0")"
ENV=${ET_ENV:-~/anaconda3/envs/executorch}
ET=$ENV/src/executorch
B=$ET/build-android
L=$B/lib
CXX=$ENV/opt/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
OUT=../app/src/main/jniLibs/arm64-v8a
mkdir -p out "$OUT"

ABSL=$(for n in log_internal_check_op leak_check raw_hash_set cord cordz_info cord_internal \
  cordz_functions cordz_handle crc_cord_state crc32c crc_internal crc_cpu_detect \
  hashtablez_sampler exponential_biased log_internal_conditions log_internal_message \
  examine_stack log_internal_format str_format_internal log_internal_proto \
  log_internal_nullguard log_internal_log_sink_set log_internal_globals log_globals hash city \
  bad_variant_access low_level_hash log_sink log_entry strerror vlog_config_internal \
  synchronization stacktrace symbolize debugging_internal demangle_internal demangle_rust \
  decode_rust_punycode utf8_for_code_point graphcycles_internal malloc_internal \
  kernel_timeout_internal time civil_time time_zone tracing_internal log_internal_fnmatch \
  strings strings_internal string_view throw_delegate base spinlock_wait int128 \
  bad_optional_access raw_logging_internal log_severity; do echo $L/libabsl_$n.a; done)
TOKLIBS=$(for n in tokenizers sentencepiece re2 regex_lookahead pcre2-8; do echo $L/lib$n.a; done)

$CXX --target=aarch64-none-linux-android30 -std=gnu++20 -O2 -fPIC -shared -DANDROID -DNDEBUG \
  -DC10_USING_CUSTOM_GENERATED_MACROS -DPCRE2_STATIC -fdata-sections -ffunction-sections \
  -Wno-deprecated-declarations \
  -I$ET/.. -I$ET/extension/llm/tokenizers/include -isystem $B/include \
  -isystem $B/include/executorch/runtime/core/portable_type/c10 -isystem $B/include/pytorch/tokenizers \
  -static-libstdc++ -Wl,--gc-sections -Wl,--no-undefined -Wl,-s -Wl,-soname,libembgemma_npu.so \
  embgemma_jni.cpp -o out/libembgemma_npu.so \
  $L/executorch/backends/qualcomm/libqnn_executorch_backend.so \
  -Wl,--whole-archive $L/libregex_lookahead.a $L/libexecutorch.a -Wl,--no-whole-archive \
  $L/libextension_module.a $L/libextension_tensor.a $L/libextension_data_loader.a \
  $L/libextension_flat_tensor.a $L/libextension_named_data_map.a $TOKLIBS \
  $L/libexecutorch_core.a $ABSL $TOKLIBS -llog -landroid -pthread -latomic -lm
cp out/libembgemma_npu.so $L/executorch/backends/qualcomm/libqnn_executorch_backend.so "$OUT/"
ls -la "$OUT"
