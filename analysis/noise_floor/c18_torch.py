"""Check 18 (torch half): write the float32 inputs of every chain, then run each chain in torch fp32 on the GPU (TF32 off) and on the CPU. Raw outputs go to
results/noise_floor/c18_chains/raw_torch_{gpu,cpu}.npz; c18_onnx.py runs the same inputs through ONNX Runtime and c18_analyse.py compares everything with theory."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis.noise_floor.chains import CHAINS, eval_torch, make_inputs

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor")) / "c18_chains"
OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
inputs = {n: make_inputs(c) for n, c in ((n, f()) for n, f in CHAINS.items())}
np.savez(OUT / "inputs.npz", **{f"{cn}/{k}": v for cn, d in inputs.items() for k, v in d.items()})
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
(OUT / "meta_torch.json").write_text(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu)))
for tag, dev, tf32 in [("gpu", "cuda", False), ("cpu", "cpu", False), ("gpu_tf32", "cuda", True)]:
    if dev == "cuda" and (gpu is None or (tf32 and torch.cuda.get_device_capability()[0] < 8)):
        continue
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    out = {}
    for cn, f in CHAINS.items():
        for lab, arr in eval_torch(f(), inputs[cn], dev).items():
            out[f"{cn}/{lab}"] = arr
    np.savez(OUT / f"raw_torch_{tag}.npz", **out)
    print("torch", tag, "done", flush=True)
