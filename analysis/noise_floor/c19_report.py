"""Check 19 (report): merge raw_torch.json and raw_onnx.json into results.json / data.npz (speedup = time with TF32 off / time with TF32 on); plots.py draws the figure."""

import json
import os
from pathlib import Path

import numpy as np

from analysis.noise_floor.common import result, save_data

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor")) / "c19_speed"
t, o = (json.loads((OUT / f"raw_{n}.json").read_text()) for n in ("torch", "onnx"))
sizes = sorted(int(n) for n in t["matmul"])
sp = lambda d: d["tf32_0"]["ms"] / d["tf32_1"]["ms"]
meas = dict(device=t["device"], torch=t["torch"], onnxruntime=o["onnxruntime"],
            matmul_speedup_torch={n: sp(t["matmul"][str(n)]) for n in sizes}, matmul_speedup_onnx={n: sp(o["matmul"][str(n)]) for n in sizes},
            matmul_tflops={f"{lib}_tf32_{k}": {n: d["matmul"][str(n)][f"tf32_{k}"]["tflops"] for n in sizes} for lib, d in (("torch", t), ("onnx", o)) for k in (0, 1)},
            block_speedup_torch={k: sp(v) for k, v in t["block"].items()}, block_speedup_onnx={k: sp(v) for k, v in o["block"].items()},
            block_ms={f"{lib}_{k}": {s: v[f"tf32_{s}"]["ms"] for s in (0, 1)} for lib, d in (("torch", t), ("onnx", o)) for k, v in d["block"].items()})
result("c19_speed", "TF32 speeds up large fp32 matmuls and blocks, but not the tiny sizes of this repo's model (launch-bound)", meas, "info", raw_torch=t, raw_onnx=o, sizes=sizes)
save_data("c19_speed", sizes=np.array(sizes))
