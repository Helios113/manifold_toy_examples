# onnx_error_bound: floating-point error bounds for ONNX graphs

## What it does
Given an ONNX model and concrete float32 inputs, it returns, for every graph output, the exact value of the program (computed in float64) and an absolute bound on how far the result of a float32 runtime (ONNX Runtime on the CPU or CUDA) can be from it. The bound covers rounding error only, not the model's own approximation error. It is data dependent: it uses the actual intermediate values, so it differs per input batch.

```python
from onnx_error_bound import estimate_bounds, load_profile

r = estimate_bounds("models/toy_model.onnx", {"x": x_float32}, load_profile("theory_fp32"))
logits, bound = r.outputs["logits"].v, r.outputs["logits"].e   # float64 reference, absolute bound (same shape)
r.warnings                                                     # e.g. an op input outside the range its profile was calibrated on
```
`estimate_bounds(model, inputs, profile=None, input_rel_error=0.0, keep_intermediates=False)`: `model` is a path or `onnx.ModelProto`; `profile` is a `Profile`, a profile name or path, or `None` (textbook fp32); `input_rel_error` adds a relative uncertainty to the inputs (default: the given float32 values are exact). Weights (initializers) are taken as exact float32 values.

## How it works
- `estimate.py` walks the graph in order. Each tensor is carried as `T(v, e)`: `v` the exact value, `e` an absolute error bound, both float64 arrays of the tensor's shape.
- `rules.py` has one rule per operator. A rule first propagates the inputs' errors (first-order rules, made rigorous with `expm1`, `max(|x| - e, 0)` and similar where it is cheap), then adds the operation's own rounding error.
  - Elementwise ops (Add, Sub, Mul, Div, Sqrt, Reciprocal, Exp, Log, Erf, Tanh, Sigmoid): rounding term `k * u * (|result| + propagated error)`; `u = 2^-24`; `k = 1` for a correctly rounded op.
  - Sum / mean over `n` elements: `sum(e) + gamma_(n-1) * sum|v|`, with `gamma_n = n u / (1 - n u)`; valid for any summation order.
  - MatMul / Gemm with inner size `n`: propagated error plus `c(n) * (|A| + eA)(|B| + eB)`, with `c(n) = gamma_n * factor` in fp32 and `c(n) = (2 u_tf32 + u_tf32^2 + gamma_n) * factor` with TF32 on (`u_tf32 = 2^-11`, both operands rounded to 11 bits first). Gemm adds the `alpha`, `beta` and bias roundings.
  - Exact ops (Neg, Abs, Relu, Identity, Reshape, Flatten, Transpose, Squeeze, Unsqueeze) pass the error through unchanged (Relu is 1-Lipschitz).
  - Fused kernels the runtime executes as one op, LayerNormalization and Gelu (`approximate="none"`), are bounded through their decomposition (mean, centre, variance, sqrt, divide, scale, bias; and `x * 0.5 * (1 + erf(x / sqrt 2))`), multiplied by a calibrated composite factor because the kernel's internal algorithm is not known.
- Unsupported ops raise `UnsupportedOp` naming the node; there is no silent fallback. Supported: Add, Sub, Mul, Div, Neg, Abs, Relu, Identity, Sqrt, Reciprocal, Exp, Log, Erf, Tanh, Sigmoid, MatMul, Gemm, ReduceSum, ReduceMean, LayerNormalization, Gelu, Reshape, Flatten, Transpose, Squeeze, Unsqueeze, Constant. The repo's exported models use only Gemm, LayerNormalization, Gelu and Add.

## Profiles: what the runtime does to the numbers
`profile.py` defines `Profile` (stored as JSON in `profiles/`). It holds `u`, the per-op multiplier `k`, the matmul mode (`fp32` or `tf32`) and factor, composite factors for LayerNormalization and Gelu, the input ranges each non-linear op was calibrated on (the estimator warns outside them), and the runtime settings (providers, provider options such as `use_tf32`, graph optimisation level).
`load_profile("theory_fp32")` and `load_profile("theory_tf32")` are built in: every op assumed correctly rounded, no composite factors. They are only valid if the runtime really rounds that well.

Why calibration is needed (from the `analysis/noise_floor` experiments, see `results/noise_floor*/c*/README.md` and `noise_floor.md`): ops such as Erf, Exp, Tanh are not correctly rounded (up to about 1.9 ulp on CUDA, fp32); ONNX Runtime's CPU Sqrt was measured at 3.3 to 3.7 ulp, not 0.5; ONNX Runtime enables TF32 on Ampere and newer GPUs by default (`use_tf32=1`), which makes a fp32 matmul about 1000 times less accurate. A profile records those facts for one backend on one device.

## Calibration (`calibrate.py`, needs onnxruntime)
```
python -m onnx_error_bound.calibrate --name ort_cuda_fp32_h100 --provider cuda --tf32 0 [--opt all|none]
```
It runs single-operator ONNX graphs through the runtime on random inputs and compares with the float64 result:
1. Elementary ops (Add, Sub, Mul, Div, Sqrt, Reciprocal, Exp, Log, Erf, Tanh, Sigmoid), 10^6 inputs each: worst error in ulp; `k = 1` if at most 0.5 ulp, else `2.5 * max ulp` (1 ulp is at most `2u` relative, plus a 25% margin).
2. MatMul at inner sizes 16, 64, 256, 1024, 4096 (normal and positive data) and a Gemm with transposed weight and bias: worst `|error| / sum|a b|` against the profile's theory coefficient; matmul factor `= max(1, 1.25 * worst ratio)`.
3. LayerNormalization (widths 16, 64, 768; mean offsets 0, 1, 10, 100 sigma) and Gelu (inputs in [-3, 3] and [-8, 8]): worst error / bound of the decomposition rule; composite factor `= max(1, 1.25 * worst ratio)`.
`run_all.sbatch` calibrates three backends on the cluster (ONNX Runtime CPU, CUDA fp32, CUDA TF32), using the custom `onnxruntime-training` build, which needs its CUDA 12 libraries on `LD_LIBRARY_PATH` for that step only.

## Validation (`validate.py`, needs onnxruntime)
```
python -m onnx_error_bound.validate --profiles ort_cpu --models m.onnx --inputs sphere=x.npy
```
Runs the model in ONNX Runtime with the profile's providers, compares with the float64 reference, and writes `validation/report.{json,md}`: the fraction of outputs inside the bound, the worst error / bound, and how loose the bound is (median bound / median error). `export_repo_models.py` exports the trained model and the two degenerate ones (DW.a, DW.b) and writes sphere and normal inputs for it. `selftest.py` is a runtime-free check (a small graph against a plain numpy float32 implementation).

## Files
`estimate.py` graph walk and public API; `rules.py` operator rules; `profile.py` profiles; `runtime.py` ONNX Runtime sessions and single-op graphs; `calibrate.py`; `validate.py`; `export_repo_models.py`; `selftest.py`; `run_all.sbatch`; `profiles/` calibrated profiles (empty until calibration is run); `validation/` reports and exported models.

## Status and limits
- **Implemented and tested:** the estimator, the rules, the profile format, and the self test (the bound held against a numpy float32 run, with the bound about 1000 times above the typical error for that tiny graph). The module needs Python 3.11 with onnx, numpy and scipy; the calibration and validation parts also need the `onnxruntime-training` wheel.
- **Not run:** the calibration and the validation were prepared and submitted once and then cancelled, so `profiles/` is empty and no calibrated profile or validation report exists yet. Until they are generated, use the textbook profiles and treat the result as valid only for a runtime that rounds like theory (the experiments above show that does not hold for ONNX Runtime's CPU Sqrt, for TF32, or for Erf-based ops).
- **Behaviour measured earlier on hand-written chains (c18):** the same rules with per-backend multipliers were never exceeded on six chains, in torch and ONNX Runtime, apart from the TF32-with-fp32-rule case which the TF32 rule fixes. The bound is a safe ceiling, often 10 to 700 times above the typical error for matmul-heavy graphs, not an estimate of the error you will see.
- **Not covered:** ops outside the list above (for example Softmax, Conv, Gelu with `approximate="tanh"`, Pow), float16/bfloat16 graphs, int tensors other than shape constants, error from the model's own approximations, and any runtime other than ONNX Runtime (a new runtime needs its own calibration, and calibration itself currently targets ONNX Runtime).
- The LayerNormalization factor is only calibrated for mean offsets up to 100 sigma and widths 16 to 768; a one-pass variance kernel would show up as a large factor.
