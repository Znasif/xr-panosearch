"""Lower the EmbeddingGemma2 text tower as ONE multi-method .pte with HTP weight sharing (V79).

Each method is the same static text embedder at a different (batch, S) shape, sized to what
PanoSearch actually runs, so no call pays for padding up to the next power-of-two bucket and the
~270 MB of weights is stored once:
  q32    [1,32]   text queries ("task: search result | query: ..." is ~12 tokens)
  grid   [9,64]   the 3x3 image sub-regions (48 soft + 4 special = 52 tokens), one batched call
  a160   [1,160]  5 s audio (<= 125 soft + 4 special)
  img288 [1,288]  full image (280 soft + 4 special)
  s512   [1,512]  long text fallback
Optionally converts every nn.Linear to a 1x1 conv2d (--conv), which HTP runs more efficiently.

Usage: python export_text_multi.py [--methods q32,grid,...] [--conv] [--soc SXR2330P] [--out X.pte]
"""
import argparse
import collections
import gc
import os
import sys

import torch

sys.path.insert(0, "/home/znasif/local/hackathon/embgemma2/text/scripts")
from static_text import StaticTextEmbedder, load_hf_text_model  # noqa: E402

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset  # noqa: E402
from executorch.backends.qualcomm.utils.utils import (  # noqa: E402
    convert_linear_to_conv2d,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import ExecutorchBackendConfig  # noqa: E402
from executorch.exir.passes import MemoryPlanningPass  # noqa: E402

SHAPES = {"q32": (1, 32), "grid": (9, 64), "a160": (1, 160), "img288": (1, 288), "s512": (1, 512)}

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default=",".join(SHAPES))
ap.add_argument("--conv", action="store_true")
ap.add_argument("--soc", default="SXR2330P")
ap.add_argument("--out", required=True)
args = ap.parse_args()
methods = args.methods.split(",")

hf = load_hf_text_model()
models, inputs = {}, {}
for name in methods:
    B, S = SHAPES[name]
    m = StaticTextEmbedder(hf.language_model, S).eval()
    if args.conv:
        m = convert_linear_to_conv2d(m)
    x = torch.randn(B, S, 512) * 0.5
    mask = torch.zeros(B, S)
    mask[:, : S // 2] = 1.0
    models[name], inputs[name] = m, (x, mask)
del hf
gc.collect()

spec = generate_qnn_executorch_compiler_spec(
    soc_model=getattr(QcomChipset, args.soc),
    backend_options=generate_htp_compiler_spec(use_fp16=True, use_weight_sharing=len(methods) > 1),
)
with torch.no_grad():
    edge = to_edge_transform_and_lower_to_qnn(models, inputs, {n: spec for n in methods})

for name in methods:
    gm = edge.exported_program(name).graph_module
    n_deleg, cpu_ops = 0, collections.Counter()
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        t = str(n.target)
        if "executorch_call_delegate" in t:
            n_deleg += 1
        elif "getitem" not in t:
            cpu_ops[t] += 1
    print(f"DELEGATION {name}: {n_deleg} delegate call(s); CPU-fallback ops: {dict(cpu_ops) or 'none'}", flush=True)

prog = edge.to_executorch(
    config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True),
        extract_delegate_segments=True,
    )
)
with open(args.out, "wb") as f:
    f.write(prog.buffer)
print(f"OK -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, soc={args.soc}, methods={methods}, conv={args.conv})")
