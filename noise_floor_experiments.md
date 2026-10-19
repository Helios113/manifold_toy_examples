# Noise-floor experiments: what we run, why, and whether it tests the right thing

Code: `analysis/noise_floor/` (`c*.py`, run by `run*.sbatch`). Results: `results/noise_floor/` (RTX 2080 Ti) and `results/noise_floor_h100/` (H100 PCIe), one folder per check with `results.json`, `plot.png` and a README; `SUMMARY.md` lists the verdicts. Literature behind each claim, with a checked/unchecked status per reference: [noise_floor.md](noise_floor.md). All numbers below are from the H100 run unless stated.

## 1. The question behind all of it

The degenerate-weight models (DW.a, DW.b) are meant to have a Jacobian that is singular, or nearly so, at some manifold points. We detect that with `σ_min(J)` and `σ_min(J|_T)`. The job-61786 log shows float64 minima of 4.8e-10 (DW.a, block 0) and 1.3e-6 (DW.b, block 0). To call a value "a real zero" or "a real small number" we need to know **the smallest σ_min that the computation can resolve**, the noise floor. Below it a value is indistinguishable from 0. Above it, it is real. So the noise-floor work has to answer:

1. How large is the rounding error in the Jacobian `J` that we compute (float64 on the CPU for the plots; float32 in the trained model and on the GPU)?
2. How much does that error move `σ_min` (Weyl: by at most `‖E‖₂`), and how much does the SVD add on top?
3. Do the assumptions that make the textbook error theory applicable hold on the hardware and runtime we use (GPU, ONNX Runtime, TF32)?

The same machinery feeds `onnx_error_bound/`, which turns per-operation theory into a data-dependent bound for a whole exported graph.

## 2. Method used in every check

Each check has the same shape. It picks a claim from the theory, builds inputs whose **exact answer is known** (exact rationals via `fractions`, integer matrices, a Hadamard construction `A = H diag(s) Hᵀ` with exactly prescribed singular values, 50-digit `mpmath`, or just float64 as reference for float32), runs the computation on the device, and compares the measured error with the predicted bound. The verdict is `holds` if the measured error never exceeds the bound. Inputs are float32 values (exactly representable in float64), so the reference sees exactly what the algorithm saw and only the algorithm's own error is measured. No model is involved; the model-based checks are parked in `analysis/noise_floor/deferred_models/` (see section 6).

## 3. The checks

| id | claim under test | how | what we found (H100) |
|---|---|---|---|
| c01 | `+ − × ÷ √` are correctly rounded: relative error ≤ u, bit-exact against the correctly rounded value | 1e6 random fp32 operands (magnitudes 1e-3 to 1e3) against fp64 then rounded; 2e4 fp64 operands against exact `Fraction` arithmetic | max error 0.9999 u, 100% bit-exact, fp32 and fp64 |
| c02 | `exp/erf/log/tanh` are **not** correctly rounded; error is a few ulp | 2e6 fp32 inputs, GPU and CPU, error in ulp against fp64 | CUDA max: exp 1.87, tanh 1.78, erf 1.01, log 0.84 ulp. CPU max 0.51 to 0.80 ulp |
| c03 | first-order propagation: `err ≤ Σ|∂f| e_in + u|result|` | inputs carry their own fp32 rounding error; 1e6 samples for add, mul, div, exp | add, mul, div: max 0.9999 of the bound. exp: exceeds it (max 10.6×; 18% of samples outside) because exp is not correctly rounded |
| c04 | dot product: `err ≤ γₙ Σ|xᵢyᵢ|`; typical `~√n u Σ|xᵢyᵢ|`; relative error to the **result** explodes under cancellation | n = 16 to 65536, normal / positive / exactly-cancelling data, `sum` and `bmm`, TF32 off | bound holds with huge slack (max 1e-6 to 0.1 of γₙ). Typical error is **not** `√n u S` (see 4.3). Cancelling data: relative-to-result error ≈ 3.9 |
| c08 | TF32 matmul has `u = 2⁻¹¹ ≈ 4.9e-4` | 256×n @ n×256 matmuls, TF32 on/off, error / Σ\|ab\| | n=16: 4.7e-4 (TF32) vs 1.9e-7 (fp32), the predicted `2u`; shrinks with n (3.4e-5 at n=4096) because errors average out |
| c09 | which GPU ops are bit-reproducible | each op repeated 100×, compared with the first run | matmul 2048² and `sum` 2e7: identical in 100/100. `index_add_` (float atomics): differs in 100/100, relative 1e-5 |
| c10 | finite-difference best step `~√u` (forward) / `u^(1/3)` (central) | `exp` derivative at 1000 points, 37 step sizes, fp32 and fp64 | best steps and errors match theory within the stated factor (fp32 forward 3.2e-4 vs 2.4e-4) |
| c13 | a computed `σ_min` is wrong by at most about `d·ε·σ_max` (d = 16) | (a) integer matrices of rank 15 and 8, so true `σ_min = 0`; (b) prescribed `σ_min = 10⁻ʲ`, j = 0..18, exact singular values of the **rounded** input from mpmath | exact zeros come out at ≤ 0.58 ε σ_max, far below 16. Sweep: max error 2.6 (fp32) and 4.25 (fp64) in units of ε σ_max, input rounding alone up to 0.43 |
| c14 | Weyl: `max_i|σ_i(A+E) − σ_i(A)| ≤ ‖E‖₂`, with equality for `E = δ q q ᵀ` | 50-digit arithmetic, 11 values of δ, adversarial and random E | adversarial ratio exactly 1, random ≤ 0.46 |
| c15 | forward error `≲ κ·u` for a backward-stable solve | LU solve on the GPU, κ = 1 to 1e16, exact solution from mpmath | max error / (κ u) = 0.45 (fp32), 0.47 (fp64), against an allowed 100 |
| c16 | `γₙ Σ|x|` is a true worst case and is attained | `[2²⁴, 1, 1, …]` summed left to right in fp32; iid U(0,1) sequential vs pairwise | adversarial case reaches 0.9999 of the bound. Random sequential error grows like √n (4e-8 to 2.2e-6), pairwise stays flat at about 3e-8 |
| c17 | a running (value, error-bound) tracer is a valid bound | Horner evaluation of random polynomials (degree 4 to 256) and dot products (n = 64, 1024), 5000 instances each | 100% inside the bound. Slack (bound / actual, median): 6 to 8 for Horner, 26 to 99 for dot products |
| c18 | the per-operation theory chains into a valid bound for whole programs, in torch and ONNX Runtime | six programs (3 matmuls, LayerNorm, GELU, the repo's residual block, 16 square roots, 64 products), 12 backend variants. The multiplier `k` of sqrt / div / erf is calibrated per backend on single-op chains | strict variants never exceed the bound (max ratio 0.02 to 1.0; the √ chain touches 1.0). With TF32 on, the fp32 bound is exceeded by up to 55× and the TF32 theory `2u_tf32 + γₙ` holds (max ratio 0.05 to 0.10) |
| c19 | (information) speed of TF32 | matmul and the residual block, N = 16 to 8192, torch and ONNX | for this repo's size TF32 gives about 1.1 to 1.3×, because the model is launch-bound. For N ≥ 1024 it gives 3 to 7× |

Unexpected backend facts found by c02 and c18: ONNX Runtime **CPU** `Sqrt` is up to 3.25 ulp off, not 0.5 (so `k = 8.1` in its profile). ONNX Runtime enables TF32 on Ampere and newer GPUs by default, which makes an fp32 matmul about 1000× less accurate (see [onnx_error_bound/README.md](onnx_error_bound/README.md)).

## 4. Is this the right way to test the theoretical expectations? An honest assessment

### 4.1 What the approach gets right

- **Exact references.** Every check uses a reference that is exact or much more accurate than the thing tested (rationals, 50-digit mpmath, constructed singular values), and it measures the matrix the algorithm actually received. That removes the usual confusion between input rounding and algorithm error.
- **Adversarial inputs next to random ones.** c16 and c14 build the worst case and show the bound is attained. A bound checked only on random data can be very loose without anyone noticing.
- **Two devices.** The same checks ran on Turing and Hopper.

### 4.2 What it does not and cannot test

- **The mathematical inequalities are theorems.** Weyl, `γₙ`, and the first-order propagation rule are proven. A numerical example cannot validate a proof, and a pass on random data adds very little to it. What these checks do test is whether the *hypotheses* hold on our hardware: IEEE correct rounding (c01), no extended or reduced precision hiding in the matmul (c08), transcendental accuracy (c02), summation order (c04, c16), reproducibility (c09). That is where the real findings came from (TF32, ORT CPU sqrt, exp/tanh at 1.9 ulp). c14, c15 and c13(a) mostly confirm that the test machinery is correct. We should say so rather than count them as evidence for the theory.
- **A bound that always holds is weak evidence.** Passing means "error ≤ bound". Many bounds hold with 10× to 10⁶× slack (c04 up to 1e6 below γₙ, c17 6 to 99×, c18 12 to 700× on matmul-heavy chains). A bound that is far too large would also pass. For our purpose a bound this loose would make every small σ_min look "unresolvable". Slack is recorded but is not part of any verdict.
- **Single seed, one input family.** Everything uses `manual_seed(0)` and Gaussian or uniform data. There is no repeat over seeds, so there is no spread, and no input resembling the model's activations (LayerNorm std of about 0.2, one-sided GELU values).
- **Calibration then test on the same distribution.** In c18 `k` for sqrt/div/erf and the TF32 matmul coefficient are taken from the maximum error on 1e6 (or 65536 matrices) of one distribution, then used to test chains with the same shapes. "The calibrated bound holds" is partly circular. The maximum of a sample is also only a lower bound on the true worst case (ORT CPU sqrt might be worse on inputs not drawn). The theory-derived TF32 bound (`2u_tf32 + u²_tf32 + γₙ`, not fitted) is the more convincing result: it holds with a 10 to 20× margin.
- **Thresholds in the verdicts are loose or arbitrary.** c15 allows a constant of 100 (observed 0.47). c10 passes if the best step is within a decade of theory and does not check the error size. c08 requires "> 100×" TF32 error. c18's √ chain reaches ratio 1.0 on CUDA, which means zero margin; I have not looked into why (a tie at the bound or a clipped histogram), so it should be treated as unexplained.

### 4.3 One claim the data contradicts

The c04 label says the typical error is about `√n u Σ|xᵢyᵢ|`. The measured ratio median / (√n u) falls from 0.06 at n=16 to 4e-5 (`sum`) and 2.3e-4 (`bmm`) at n=65536. So the **typical error of the GPU kernels is far below that rule**. The reasons are visible in c16: GPU reductions are blocked or pairwise (error flat in n, about 3e-8), not sequential (error grows like √n), and for random-sign data the error scales with the partial sums, which are about √n, not with Σ|xᵢyᵢ| ~ n. The verdict is still `holds` because it only checks the worst-case bound. The README/SUMMARY wording should say "worst-case bound holds; typical error is much smaller on GPU kernels".

### 4.4 Gaps between the checks and the question we actually care about

1. **We bound a computed SVD of an exact matrix, but our matrix is not exact.** c13 shows the SVD adds at most about 4 ε σ_max. The dominant term is the error `E` in the entries of `J` itself, from evaluating the analytic Jacobian (LayerNorm Jacobian, GELU derivative, two matmuls) in float64 or float32. No check bounds the error of the **Jacobian formula**. c17 and c18 cover the forward pass, not the Jacobian. The overall floor is `‖E_J‖₂ + SVD error`, and only the second term has been measured.
2. **float64 transcendental accuracy is untested.** The plots use float64 for J, and GELU' needs `erf` and `exp`. c02 measures fp32 only; c01 covers fp64 basic ops only. The float64 floor of 1e-14 quoted in [noise_floor.md](noise_floor.md) item 17 assumes fp64 `erf`/`exp` are accurate to a few ulp. That is likely true, but we have no measurement.
3. **The decisive model-based checks are parked.** `deferred_models/` holds c05 (Weyl on the real J), c06 (float32 vs float64 SVD), c07 (float64 floor), c11 (Monte Carlo perturbation) and c12 (sampling resolution), plus `c00_models.py`. These are the ones that answer "is `σ_min = 4.8e-10` at point 0 of DW.a real?". The analytic checks justify the method. Only the model checks decide the question. The job-61786 log gives the float64 floor of about 8e-15 and float32 floor of about 5e-6, but that uses the heuristic `σ_max · n · ε`, not a measured error of J.
3a. **A single-point minimum is not an infimum.** `min σ_min` over 6000 samples only bounds the true minimum from above (noise_floor.md item 16, my own argument, with `L` unmeasured). Resolution matters as much as rounding for DW.a, which is a one-point degeneracy.
4. **The hardware relevant to the models was not always the one tested.** The trained models are tiny and launch-bound; c19 shows TF32 would not even help them. c08 and c18's TF32 variants need Ampere or newer, so the 2080 Ti run has no TF32 data. That is fine as long as TF32 is off for the model, but nothing in the repo asserts it. The ORT default of `use_tf32=1` is the risk.
5. **Reproducibility (c09) tests three ops once.** Matmul and sum were deterministic at a single size and stream. That does not prove cuBLAS is deterministic for the model's shapes or other workspace settings.

### 4.5 Verdict

For what it set out to do, **checking that the textbook floating-point model describes this hardware and runtime**, the design is sound: exact references, worst-case and typical cases, two GPUs, and it already found three real deviations (transcendental ulp, ORT CPU sqrt, TF32). For the question "is σ_min at this point numerically zero?" it is **necessary but not sufficient**. The remaining weakness is not more analytic checks, but that nothing yet ties an error bound for the *Jacobian of the actual model* to the measured `σ_min`.

## 5. What I would change (and only this)

Ordered by how much each changes a conclusion:

1. Add the missing measurement for gap 4.4.1: for the DW and trained models, compute J in float64 and in an exact or higher-precision reference (mpmath at a few points, including DW.a's point 0 and DW.b's worst points) and report `‖J − J_exact‖₂` next to `σ_min`. This replaces the heuristic floor with a measured one.
2. Add an fp64 `exp`/`erf` ulp measurement to c02 (closes 4.4.2).
3. Re-run the parked c05/c06/c11 on the real J, or state that the exact-J measurement in item 1 replaces them.
4. Correct the c04 label and verdict to separate "worst-case bound holds" from "typical error is much smaller".
5. Repeat the random checks over several seeds and report the spread. Put the slack (bound / actual) into the verdict criteria for c17, c18, so "holds" also says how tight.

## 6. Status

- Run and recorded: c01 to c04, c08 to c10, c13 to c19 on both GPUs (c08 and the TF32 rows of c18 on the H100 only; c19 on the H100 only).
- Parked and not run: c05, c06, c07, c11, c12 (`analysis/noise_floor/deferred_models/`). The numbering gaps in the results folders are these checks.
- References: several are still marked unconfirmed in [noise_floor.md](noise_floor.md) (Higham section numbers, Nocedal & Wright, Verrou, Weyl 1912).
