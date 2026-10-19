# c18 chains: predicted error bound vs measured error through chains of operations

Figures: `plot.png` (CDF of error / bound per chain and backend) and `plot_bounds.png` (median bound vs median error). Two runs: `results/noise_floor/c18_chains/` (ngongotaha, RTX 2080 Ti: torch GPU + CPU, ONNX Runtime CPU) and `results/noise_floor_h100/c18_chains/` (ruapehu, H100: everything, including ONNX Runtime CUDA).

## What we test
That the error bound obtained by chaining the per-operation rules (c01 to c04) through a whole computation is not exceeded by the measured error, in torch and in an ONNX Runtime graph, and which settings break it.

## How
- Six chains, each defined once in a small IR (`analysis/noise_floor/chains.py`) and run as torch ops and as an ONNX graph with the same op sequence (opset 17; ReduceSum for the sums): 3 chained 16x16 matmuls; LayerNorm (sum, mul, sub, sqrt, div); GELU (erf, mul, add); the full residual MLP block (LayerNorm, matmul 16x64, bias, GELU, matmul 64x16, bias, residual); 16 chained sqrt; 64 chained multiplications. Batch 4096, float32 inputs and weights, width 16, hidden 64.
- Reference: the same program in float64 (scipy `erf`) on the same float32 inputs. Measured error = |output - reference| per output element.
- Prediction: `trace()` in `chains.py`. add/sub/mul/div/sqrt/erf by the c03 rules, matmul and sum by `gamma_n` (valid for any summation order). The rounding-term multiplier `k` of sqrt, div and erf is calibrated per backend from a single-op run (1e6 inputs): `k = 1` if the worst error is <= 0.5 ulp, otherwise `2.5 * max ulp` (1 ulp <= 2u relative, plus a 25% margin).
- Backends: torch GPU (TF32 off), torch CPU, ONNX Runtime 1.20 training build (`cu126`, sm86) on CPU and on CUDA, each with graph optimisations off and on, plus CUDA with `use_tf32=1` (ONNX Runtime's default on Ampere and newer).

## What the literature says
The same standard model and propagation rules as c01, c03, c04 and c17 (Higham, Ch. 2 and 3; running error analysis), with a guaranteed bound. Expected: the bound holds for every output, is loose by an amount that grows with chain length (c17), and is violated only where an operation has more error than assumed (TF32, uncalibrated approximations).

## How we use this later
Measured on the H100 (ruapehu); the ngongotaha run gives the same torch and ONNX CPU numbers.

| backend | bound holds in all six chains | note |
|---|---|---|
| torch GPU, torch CPU | yes | `k` for erf 2.5 (GPU) / 1.6 (CPU), sqrt and div 1 (torch CPU sqrt 1.9) |
| ONNX Runtime CPU, with or without fusion | yes | **CPU sqrt is not correctly rounded: 3.3 to 3.7 ulp, so `k` = 8 to 9** |
| ONNX Runtime CUDA, with or without fusion | yes | identical to torch GPU, `k` for erf 2.5 |
| ONNX Runtime CUDA or torch GPU, TF32 on, **fp32 bound** | **no** | matmul chain: 6.9% of outputs inside the bound, worst 55 times the bound; MLP block: 14.6% inside, worst 26 times (predicted instead with the TF32 rule below) |

- Fusion changed nothing on this model: the fused and unfused graphs gave identical errors.
- Looseness (median bound / median error): 500 for 3 matmuls, 660 to 720 for the MLP block, 70 for LayerNorm, 12 for GELU, 4 to 23 for 16 sqrt, 17.5 for 64 multiplications. The long matmul chains are loose because `gamma_n` is a worst-case bound and the errors average out.
- So a bound traced with the c01 to c04 rules plus a calibrated `k` per backend is safe for torch and ONNX Runtime on CPU and CUDA, but the calibration must be redone for every backend: a plain `k = 1` for sqrt would have been wrong for ONNX Runtime on the CPU. ONNX Runtime with TF32 on (its default on Ampere and newer) is outside this model: either switch it off (`use_tf32=0`) or use the TF32 unit roundoff 2^-11 for the matmuls.
- The ONNX Runtime CUDA provider in this wheel is built for sm86. It failed on the RTX 2080 Ti (`no kernel image is available`) and ran on the H100, presumably through JIT-compiled PTX. It was not tested on an A40.

## TF32 prediction (added run, H100)
Question: can the `use_tf32=1` case (ONNX Runtime) and torch with `allow_tf32=True` be predicted? Figure: `plot_tf32.png`.
- **Theory:** with TF32 both matmul operands are rounded to 11 bits (10 stored mantissa bits), so each product has relative error up to `2 u_tf32 + u_tf32^2` (`u_tf32 = 2^-11`) on top of the fp32 accumulation error `gamma_n`. The tracer's matmul rule becomes `(2 u_tf32 + u_tf32^2 + gamma_n) * sum|a_i b_i|` over the computed operands (assumes the conversion rounds to nearest).
- **Calibrated alternative:** `1.25 *` the worst `|error| / sum|a_i b_i|` of one matmul with the same inner size (16 or 64), measured by the two calibration chains.
- **Result:** both hold for every output of both chains that contain matmuls, in both libraries (the plain fp32 bound fails there, as before). Max error / bound: theory 0.05 (3 matmuls) and 0.10 (block); calibrated 0.08 and 0.16. Bound / actual error (median): theory 133 and 77; calibrated 87 and 46.
- Measured worst single-matmul error / `sum|a_i b_i|` with TF32: 5.2e-4 (inner size 16) and 2.9e-4 (inner size 64), against 2.9e-7 and 3.4e-7 without TF32 (about 1800 and 870 times larger). The theory coefficient `2 u_tf32` = 9.8e-4 is about twice the measured 5.2e-4, so round-to-nearest is the right model and no truncation penalty was needed.
- torch and ONNX Runtime give identical TF32 results here (same cuBLAS kernels), so they are one curve in the plot.
- **How to use it:** for a graph run with TF32 on, replace `gamma_n` by `2 u_tf32 + u_tf32^2 + gamma_n` in every matmul of the tracer. The bound is safe and about 50 to 130 times above the typical error for these chains; the calibrated coefficient is a bit tighter.
