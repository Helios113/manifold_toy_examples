"""Check 18 (ONNX half). Builds each chain as an ONNX graph (opset 17, one node per IR node, so the op sequence is the same as torch's) and runs the inputs written by c18_torch.py
in ONNX Runtime (custom onnxruntime-training 1.20 build, sm86): CPU and CUDA execution providers, graph optimisations off (the faithful op sequence) and on (fusions), and
CUDA with TF32 enabled (ORT's default on Ampere). Runs in the main (Python 3.11) environment, which has the onnxruntime-training wheel; the CUDA libraries it was built against must be on LD_LIBRARY_PATH (see run_chains.sbatch)."""

import json
import os
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort

from analysis.noise_floor.chains import CHAINS
from analysis.noise_floor.onnx_build import build

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor")) / "c18_chains"
CUDA = lambda tf32: [("CUDAExecutionProvider", {"use_tf32": tf32}), "CPUExecutionProvider"]
LVL = ort.GraphOptimizationLevel
VARIANTS = {"ort_cpu_opt0": (["CPUExecutionProvider"], LVL.ORT_DISABLE_ALL), "ort_cpu_optall": (["CPUExecutionProvider"], LVL.ORT_ENABLE_ALL),
            "ort_cuda_opt0": (CUDA(0), LVL.ORT_DISABLE_ALL), "ort_cuda_optall": (CUDA(0), LVL.ORT_ENABLE_ALL), "ort_cuda_tf32": (CUDA(1), LVL.ORT_DISABLE_ALL)}
raw_in = np.load(OUT / "inputs.npz")
meta = dict(onnxruntime=ort.__version__, available=ort.get_available_providers(), variants={})
for var, (providers, level) in VARIANTS.items():
    try:
        out = {}
        for cn, f in CHAINS.items():
            c = f()
            inp = {n: raw_in[f"{cn}/{n}"] for n in c.inputs}
            so = ort.SessionOptions()
            so.graph_optimization_level = level
            so.log_severity_level = 3
            sess = ort.InferenceSession(build(c, inp), so, providers=providers)
            used = sess.get_providers()
            if providers[0] != "CPUExecutionProvider" and used[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"CUDA provider not active, session uses {used}")
            res = sess.run(list(c.outputs.values()), {n: np.ascontiguousarray(inp[n], dtype=np.float32) for n in c.inputs})
            for lab, arr in zip(c.outputs, res):
                out[f"{cn}/{lab}"] = arr
        np.savez(OUT / f"raw_{var}.npz", **out)
        meta["variants"][var] = dict(ok=True, providers=used)
        print("ort", var, "done", used, flush=True)
    except Exception as e:  # keep going: report which variants could not run
        meta["variants"][var] = dict(ok=False, error=repr(e), trace=traceback.format_exc()[-600:])
        print("ort", var, "FAILED", repr(e), flush=True)
    (OUT / "meta_onnx.json").write_text(json.dumps(meta, indent=1))  # after every variant, in case a later (CUDA) one kills the process
