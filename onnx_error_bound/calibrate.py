"""Calibrate a runtime backend and write its profile (profiles/<name>.json). Single-operator ONNX graphs are run through ONNX Runtime on random inputs and compared with the exact
(float64) result:
  1. every elementary op (Add Sub Mul Div Sqrt Reciprocal Exp Log Erf Tanh Sigmoid): worst error in ulp -> rounding multiplier k;
  2. MatMul and Gemm at inner sizes 16 ... 4096: worst |error| / sum|a b| against the profile's theory coefficient (gamma_n, or the TF32 rule) -> matmul factor;
  3. the fused kernels LayerNormalization (widths 16, 64, 768; mean offsets 0 ... 100 sigma) and Gelu: worst error / bound of the decomposition rule -> composite factor.

    python -m onnx_error_bound.calibrate --name ort_cuda_fp32_h100 --provider cuda --tf32 0
"""

import argparse
import json

import numpy as np
import onnxruntime as ort
from scipy.special import erf as _erf, expit

from .estimate import estimate_bounds
from .profile import Profile
from .runtime import run, single_op_model

REF = {"Sqrt": np.sqrt, "Reciprocal": lambda x: 1.0 / x, "Exp": np.exp, "Log": np.log, "Erf": _erf, "Tanh": np.tanh, "Sigmoid": expit,
       "Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
RANGE = {"Sqrt": (1e-3, 1e3), "Reciprocal": (0.1, 10), "Exp": (-10, 10), "Log": (1e-3, 1e3), "Erf": (-4, 4), "Tanh": (-5, 5), "Sigmoid": (-10, 10),
         "Add": (-1e3, 1e3), "Sub": (-1e3, 1e3), "Mul": (-1e3, 1e3), "Div": (0.1, 10)}
BINARY = {"Add", "Sub", "Mul", "Div"}


def elementary(P, n, rng):
    k, rng_info, worst = {}, {}, {}
    for op, (lo, hi) in RANGE.items():
        if op in BINARY:
            sh = {"a": (n, 1), "b": (n, 1)}
            ins = {"a": rng.uniform(lo, hi, (n, 1)).astype(np.float32), "b": rng.uniform(lo, hi, (n, 1)).astype(np.float32)}
            if op == "Div":
                ins["b"] = rng.uniform(0.1, 10, (n, 1)).astype(np.float32)
            ref = REF[op](ins["a"].astype(np.float64), ins["b"].astype(np.float64))
        else:
            sh = {"x": (n, 1)}
            ins = {"x": rng.uniform(lo, hi, (n, 1)).astype(np.float32)}
            ref = REF[op](ins["x"].astype(np.float64))
        out = run(single_op_model(op, sh, (n, 1)), P, ins)[0].astype(np.float64)
        ulp = np.spacing(ref.astype(np.float32)).astype(np.float64)
        keep = np.abs(ref) > 1e-30
        mu = float((np.abs(out - ref)[keep] / ulp[keep]).max())
        worst[op] = mu
        if op in ("Sqrt", "Exp", "Log", "Erf", "Tanh", "Sigmoid", "Reciprocal", "Div"):
            k[op] = 1.0 if mu <= 0.5 + 1e-9 else 2.5 * mu
        rng_info[op] = [lo, hi]
        print(f"  {op:10s} max error {mu:.3f} ulp -> k = {k.get(op, 1.0):.2f}", flush=True)
    return k, rng_info, worst


def matmuls(P, rng):
    measured, ratio = {}, []
    for n in [16, 64, 256, 1024, 4096]:
        worst = 0.0
        for kind in ("normal", "positive"):
            A, B = rng.standard_normal((1024, n)), rng.standard_normal((n, 16))
            if kind == "positive":
                A, B = np.abs(A), np.abs(B)
            A32, B32 = A.astype(np.float32), B.astype(np.float32)
            out = run(single_op_model("MatMul", {"a": A32.shape, "b": B32.shape}, (1024, 16)), P, {"a": A32, "b": B32})[0].astype(np.float64)
            a64, b64 = A32.astype(np.float64), B32.astype(np.float64)
            worst = max(worst, float((np.abs(out - a64 @ b64) / (np.abs(a64) @ np.abs(b64))).max()))
        theory = P.matmul_coef(n)
        measured[n] = worst
        ratio.append(worst / theory)
        print(f"  MatMul n={n:5d}: worst |err| / sum|ab| = {worst:.2e}  (theory {theory:.2e}, ratio {worst / theory:.3f})", flush=True)
    # Gemm with transposed weight and bias, as torch exports it
    x = rng.standard_normal((1024, 64)).astype(np.float32)
    W, bvec = rng.standard_normal((16, 64)).astype(np.float32), rng.standard_normal(16).astype(np.float32)
    model = single_op_model("Gemm", {"a": x.shape, "b": W.shape, "c": bvec.shape}, (1024, 16), transB=1)
    out = run(model, P, {"a": x, "b": W, "c": bvec})[0].astype(np.float64)
    r = estimate_bounds(model, {"a": x, "b": W, "c": bvec}, P).outputs["y"]
    gemm_ratio = float((np.abs(out - r.v) / r.e).max())
    print(f"  Gemm(transB, bias) n=64: worst error / bound = {gemm_ratio:.3f}", flush=True)
    return measured, max(ratio), gemm_ratio


def composites(P, rng):
    fac = {}
    ratios = []
    for width in (16, 64, 768):
        for off in (0.0, 1.0, 10.0, 100.0):
            x = (rng.standard_normal((2048, width)) + off).astype(np.float32)
            sc, bi = rng.standard_normal(width).astype(np.float32), rng.standard_normal(width).astype(np.float32)
            model = single_op_model("LayerNormalization", {"x": x.shape, "s": sc.shape, "b": bi.shape}, x.shape, axis=-1, epsilon=1e-5)
            out = run(model, P, {"x": x, "s": sc, "b": bi})[0].astype(np.float64)
            r = estimate_bounds(model, {"x": x, "s": sc, "b": bi}, P).outputs["y"]
            ratios.append(float((np.abs(out - r.v) / r.e).max()))
            print(f"  LayerNormalization width {width:4d} mean offset {off:5.0f} sigma: worst error / bound = {ratios[-1]:.3f}", flush=True)
    fac["LayerNormalization"] = max(1.0, 1.25 * max(ratios))
    ratios = []
    for lo in (3, 8):
        x = rng.uniform(-lo, lo, (2048, 512)).astype(np.float32)
        model = single_op_model("Gelu", {"x": x.shape}, x.shape, approximate="none")
        out = run(model, P, {"x": x})[0].astype(np.float64)
        r = estimate_bounds(model, {"x": x}, P).outputs["y"]
        ratios.append(float((np.abs(out - r.v) / r.e).max()))
        print(f"  Gelu x in [-{lo}, {lo}]: worst error / bound = {ratios[-1]:.3f}", flush=True)
    fac["Gelu"] = max(1.0, 1.25 * max(ratios))
    return fac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--provider", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--tf32", type=int, default=0)
    ap.add_argument("--opt", choices=["all", "none"], default="all")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cuda = a.provider == "cuda"
    P = Profile(name=a.name, runtime=f"onnxruntime {ort.__version__}", opt_level=a.opt,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"],
                provider_options={"CUDAExecutionProvider": {"use_tf32": a.tf32}} if cuda else {}, notes=f"calibrated with seed {a.seed}, n = {a.n}")
    if cuda:
        import subprocess

        P.device = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    else:
        P.device = "cpu"
    P.matmul["mode"] = "tf32" if (cuda and a.tf32) else "fp32"
    print(f"== {a.name}: {P.device}, {P.runtime}, matmul mode {P.matmul['mode']}", flush=True)
    P.k, P.ranges, worst_ulp = elementary(P, a.n, rng)
    measured, worst_ratio, gemm_ratio = matmuls(P, rng)
    P.matmul["measured"] = {str(n): v for n, v in measured.items()}
    P.matmul["factor"] = max(1.0, 1.25 * worst_ratio, 1.25 * gemm_ratio)
    P.composite = composites(P, rng)
    P.ranges["LayerNormalization"] = None
    P.ranges["Gelu"] = [-8.0, 8.0]
    P.ranges = {k: v for k, v in P.ranges.items() if v}
    P.notes += f"; worst elementary ulp errors: { {k: round(v, 3) for k, v in worst_ulp.items()} }; LayerNormalization calibrated for mean offsets up to 100 sigma, widths 16/64/768"
    print("saved", P.save(), json.dumps(dict(k=P.k, matmul_factor=P.matmul["factor"], composite=P.composite)), flush=True)


if __name__ == "__main__":
    main()
