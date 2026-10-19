"""Check 18 (analysis half). For every backend variant (torch GPU/CPU, ONNX Runtime CPU/CUDA, with and without graph optimisations, CUDA with TF32):
1. calibrate the rounding-term multiplier k of sqrt / div / erf from the single-op chains (k = 1 if the worst error is <= 0.5 ulp, else 2.5 * max ulp: 1 ulp <= 2u relative, plus a 25% margin);
2. predict the error bound of each chain with the tracer (chains.trace), using that variant's k;
3. compare with the measured error |output - float64 reference| on the same float32 inputs.
Writes results/noise_floor/c18_chains/{results.json,data.npz}; plots.py draws the figures."""

import json
import os
from pathlib import Path

import numpy as np

from analysis.noise_floor.chains import CAL, CHAINS, TEST, eval_numpy, gamma, trace
from analysis.noise_floor.common import result, save_data

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor")) / "c18_chains"
inputs = np.load(OUT / "inputs.npz")
variants = sorted(p.stem[4:] for p in OUT.glob("raw_*.npz"))
meta = {**json.loads((OUT / "meta_torch.json").read_text()), "onnx": json.loads((OUT / "meta_onnx.json").read_text()) if (OUT / "meta_onnx.json").exists() else None}
ref, chains = {}, {}
for cn, f in CHAINS.items():
    chains[cn] = f()
    ref[cn] = eval_numpy(chains[cn], {n: inputs[f"{cn}/{n}"] for n in chains[cn].inputs})
bins = np.linspace(-6, 1, 141)
UT = 2.0**-11  # TF32 unit roundoff: 10 explicit mantissa bits
res, arrays = {}, {}
for var in variants:
    raw = np.load(OUT / f"raw_{var}.npz")
    k, cal = {}, {}
    for cn, key in [("cal_erf", "erf"), ("cal_sqrt", "sqrt"), ("cal_div", "div")]:
        r = ref[cn][chains[cn].outputs["y"]]
        ulp = np.spacing(r.astype(np.float32)).astype(np.float64)
        mu = float((np.abs(raw[f"{cn}/y"].astype(np.float64) - r) / ulp).max())
        k[key] = 1.0 if mu <= 0.5 + 1e-9 else 2.5 * mu
        cal[key] = dict(max_ulp=mu, k=k[key])
    mm = {}  # worst |error| / sum|a_i b_i| of one matmul, per inner dimension
    for n, cn in [(16, "cal_mm16"), (64, "cal_mm64")]:
        c = chains[cn]
        xs, W = (ref[cn][i] for i in c.inputs)
        mm[n] = float((np.abs(raw[f"{cn}/y"].astype(np.float64) - ref[cn][c.outputs["y"]]) / (np.abs(xs) @ np.abs(W))).max())
    cal["matmul_worst_over_sum_abs"] = {str(n): v for n, v in mm.items()}
    # the fp32 bound for everything; for TF32 backends also the TF32 theory (inputs rounded to 11 bits: 2 u_tf32 + u_tf32^2 + gamma_n) and the calibrated bound (1.25 x measured worst)
    modes = {"": k}
    if var.endswith("_tf32"):
        modes["+th"] = {**k, "mm_coef": lambda n: 2 * UT + UT**2 + gamma(n)}
        modes["+cal"] = {**k, "mm_coef": lambda n: 1.25 * mm[n]}
    for mode, kk in modes.items():
        key = var + mode
        res[key] = dict(calibration=cal, chains={})
        for cn in TEST:
            c, vals = chains[cn], ref[cn]
            e = trace(c, vals, kk)
            errs, bnds = [], []
            for lab, node in c.outputs.items():
                errs.append(np.abs(raw[f"{cn}/{lab}"].astype(np.float64) - vals[node]).ravel())
                bnds.append(e[node].ravel())
            err, bnd = np.concatenate(errs), np.concatenate(bnds)
            ratio = err / bnd
            arrays[f"hist__{key}__{cn}"] = np.histogram(np.log10(np.clip(ratio, 1e-6, 10 - 1e-9)), bins=bins)[0]
            arrays[f"over__{key}__{cn}"] = np.array(int((ratio >= 10).sum()))
            res[key]["chains"][cn] = dict(frac_within_bound=float((ratio <= 1).mean()), max_err_over_bound=float(ratio.max()), median_err_over_bound=float(np.median(ratio)),
                                          median_bound=float(np.median(bnd)), median_err=float(np.median(err)), max_err=float(err.max()), max_bound=float(bnd.max()), n=int(err.size))
arrays["bins"] = bins
save_data("c18_chains", **arrays)
strict = [v for v in variants if not v.endswith("_tf32")]
within = lambda keys: all(res[v]["chains"][cn]["frac_within_bound"] == 1.0 for v in keys for cn in TEST)
tf32 = {v: dict(fp32_bound_holds=within([v]), tf32_theory_holds=within([v + "+th"]), tf32_calibrated_holds=within([v + "+cal"])) for v in variants if v.endswith("_tf32")}
result("c18_chains", "chains of operations: the error bound predicted from per-operation theory (with calibrated k for sqrt/div/erf) is not exceeded, in torch and in ONNX Runtime; "
       "with TF32 on, the TF32 theory (2 u_tf32 + gamma_n per matmul) and a calibrated matmul coefficient are tested too",
       res, "holds" if within(strict) else "violated", variants=variants, tf32_modes=[v + m for v in variants if v.endswith("_tf32") for m in ("+th", "+cal")], tf32=tf32, chains=list(TEST), meta=meta)
