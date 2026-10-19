"""Check 19 (torch half): speed of fp32 matmul and of the residual MLP block with TF32 off and on. Square matmul NxN for N = 16 ... 8192, and the block chain (chains.block) at three sizes
(batch, width, hidden): S = (4096, 16, 64) is this repo's model, M = (4096, 256, 1024), L = (16384, 1024, 4096). cuda-event timing, warm-up, median of 5 loops. Writes results/noise_floor/c19_speed/raw_torch.json."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis.noise_floor.chains import block, make_inputs

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor")) / "c19_speed"
OUT.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, "TF32 needs an Ampere or newer GPU"


def bench(fn, iters, loops=5, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(loops):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b) / iters)
    return float(np.median(ms))


def run_nodes(c, env):
    f = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b, "matmul": torch.matmul,
         "sum": lambda a: a.sum(-1, keepdim=True), "sqrt": torch.sqrt, "erf": torch.erf}
    e = dict(env)
    for out, kind, ins in c.nodes:
        e[out] = f[kind](*[e[i] for i in ins])
    return e


res = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, matmul={}, block={})
for N in [16, 64, 256, 1024, 2048, 4096, 8192]:
    A, B = torch.randn(N, N, device="cuda"), torch.randn(N, N, device="cuda")
    iters = int(np.clip(5e12 / (2 * N**3), 20, 5000))
    for tf in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = tf
        ms = bench(lambda: A @ B, iters)
        res["matmul"].setdefault(str(N), {})[f"tf32_{int(tf)}"] = dict(ms=ms, tflops=2 * N**3 / ms / 1e9)
    print("matmul", N, res["matmul"][str(N)], flush=True)
for name, (b, d, h) in {"S": (4096, 16, 64), "M": (4096, 256, 1024), "L": (16384, 1024, 4096)}.items():
    c = block(b, d, h)
    env = {k: torch.tensor(v, device="cuda") for k, v in make_inputs(c).items()}
    iters = {"S": 500, "M": 200, "L": 50}[name]
    for tf in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = tf
        res["block"].setdefault(name, dict(size=[b, d, h]))[f"tf32_{int(tf)}"] = dict(ms=bench(lambda: run_nodes(c, env), iters))
    print("block", name, res["block"][name], flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
(OUT / "raw_torch.json").write_text(json.dumps(res, indent=1))
