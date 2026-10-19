"""Check 19 (ONNX Runtime half): the same timings through ONNX Runtime's CUDA provider with use_tf32 = 0 and 1 (all graph optimisations on, as in production): a MatMul graph of NxN
and the block graph at sizes S, M, L. Inputs are bound on the GPU (IO binding) so no host copies are timed. Writes raw_onnx.json next to the torch one."""

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from analysis.noise_floor.chains import Chain, block, make_inputs, normal
from analysis.noise_floor.onnx_build import build

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor")) / "c19_speed"
OUT.mkdir(parents=True, exist_ok=True)


def session(model, tf32):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(model, so, providers=[("CUDAExecutionProvider", {"use_tf32": tf32}), "CPUExecutionProvider"])


def bench(sess, c, inp, iters, loops=5, warmup=20):
    io = sess.io_binding()
    keep = []
    for n in c.inputs:
        v = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(inp[n]), "cuda", 0)
        keep.append(v)
        io.bind_ortvalue_input(n, v)
    for o in c.outputs.values():
        io.bind_output(o, "cuda")
    for _ in range(warmup):
        sess.run_with_iobinding(io)
    ms = []
    for _ in range(loops):
        t = time.perf_counter()
        for _ in range(iters):
            sess.run_with_iobinding(io)
        ms.append((time.perf_counter() - t) * 1e3 / iters)  # run_with_iobinding returns after the CUDA stream has finished
    return float(np.median(ms))


res = dict(onnxruntime=ort.__version__, matmul={}, block={})
for N in [16, 64, 256, 1024, 2048, 4096, 8192]:
    c = Chain("mm")
    c.outputs = {"y": c.op("matmul", c.inp("a", (N, N), normal()), c.inp("b", (N, N), normal()))}
    inp = make_inputs(c)
    model = build(c, inp)
    iters = int(np.clip(5e12 / (2 * N**3), 20, 5000))
    for tf in [0, 1]:
        ms = bench(session(model, tf), c, inp, iters)
        res["matmul"].setdefault(str(N), {})[f"tf32_{tf}"] = dict(ms=ms, tflops=2 * N**3 / ms / 1e9)
    print("matmul", N, res["matmul"][str(N)], flush=True)
for name, (b, d, h) in {"S": (4096, 16, 64), "M": (4096, 256, 1024), "L": (16384, 1024, 4096)}.items():
    c = block(b, d, h)
    inp = make_inputs(c)
    model = build(c, inp)
    iters = {"S": 500, "M": 200, "L": 50}[name]
    for tf in [0, 1]:
        res["block"].setdefault(name, dict(size=[b, d, h]))[f"tf32_{tf}"] = dict(ms=bench(session(model, tf), c, inp, iters))
    print("block", name, res["block"][name], flush=True)
(OUT / "raw_onnx.json").write_text(json.dumps(res, indent=1))
