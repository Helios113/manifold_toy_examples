# Claims register: what the noise-floor bound relies on, and how each claim is verified

Purpose: the final guarantee must not rest on "a search said so". Every claim that enters the bound is written here with its exact hypotheses, a class, and a way to check it that does not need our word. Sources are listed for credit and for locating more detail, never as the only justification.

Companion files: [noise_floor.md](noise_floor.md) (literature notes, with per-reference confirmation status), [noise_floor_experiments.md](noise_floor_experiments.md) (what each experiment tests), `analysis/noise_floor/` (code), `results/noise_floor*/` (results).

## How to read this

Class decides how a claim can be verified:

- **T (theorem).** True by mathematics under stated hypotheses. Verified by reading the proof written below. A numerical example cannot prove it. Our experiments only confirm that the hypotheses hold on the hardware.
- **H (hardware/library fact).** A statement about how a specific device, driver or library behaves. It can only be falsified, never proven, and holds only for the pinned stack (section 5). Verified by a test designed to break it, plus vendor documentation.
- **R (rule of thumb).** Not a guarantee. Must not enter the final bound.

Columns "Checked by / date" are for a colleague to fill in after reading the proof or the source. **Nobody has signed off any row yet.** The status column says what *I* have done: `proof below` (I wrote it, unreviewed), `tested` (an experiment exists), `source unseen` (the book or standard was not read by me; I only saw secondary citations).

Notation: `u = 2⁻²⁴` fp32 / `2⁻⁵³` fp64 (round to nearest), `ε = 2u`. `fl(·)` is the computed value. "Normal range" means no overflow, no underflow, no subnormals. `γₙ = nu/(1−nu)`, valid for `nu < 1`.

## 1. Which claims the final bound uses

The bound on the smallest singular value of the Jacobian `J` at a point is assembled as

```
|σ_min(J_computed) − σ_min(J_true)|  ≤  ‖E_J‖₂ + δ_SVD
```

- `E_J` = error in the entries of the computed `J`. Bounded entry by entry with the running error analysis (T1, T2, T3, using H1, H3, and H7 for float64).
- First term of the sum is turned into a σ error by Weyl (T4).
- `δ_SVD` = error added by the SVD routine. Bounded by H6, or removed altogether by recomputing the few critical points in exact arithmetic (section 4).

Everything else in the table (c10 finite differences, the `κu` rule, Monte Carlo arithmetic) is background and is not used.

## 2. Register

### Theorems (class T)

| id | statement (with hypotheses) | proof | source (locator) | status | checked by / date |
|---|---|---|---|---|---|
| T0 | **Round-to-nearest error, both forms.** In the normal range, `fl(x) = x(1+δ₁)` and `fl(x) = x/(1+δ₂)`, with `|δᵢ| ≤ u`. Equivalently `|fl(x) − x| ≤ u·|x|` and `≤ u·|fl(x)|`. | §3.0 | Higham, ch. 2 (section number unconfirmed) | proof below; source unseen | |
| T1 | **Dot product / matmul entry.** For recursive or any-order summation of `n` products (with or without FMA), `|fl(xᵀy) − xᵀy| ≤ γₙ Σ|xᵢyᵢ|`, `nu < 1`, given H1 for each operation. | §3.1 | Higham, ch. 3, Lemma 3.1 (numbers unconfirmed). `γₙ` form checked at FLAME notes and Ipsen's slides | proof below; tested c04, c16 | |
| T2 | **Propagation through one operation.** If inputs `â, b̂` satisfy `|â−a| ≤ eₐ`, `|b̂−b| ≤ e_b`, then the computed result's error is at most the propagated term plus the operation's own rounding, per the table in §3.2 (add, sub, mul, div, sqrt, erf, sum, matmul). | §3.2 | Wilkinson 1963 (running error analysis); Higham §3.3 (unconfirmed) | proof below for rigorous forms; **two rules in the code are not the rigorous form**, see §3.2 | |
| T3 | **Summation.** `n` terms in any order: `|fl(Σxᵢ) − Σxᵢ| ≤ γ_{n−1} Σ|xᵢ|`. | §3.1 (same argument as T1) | Higham ch. 4 (unconfirmed) | proof below; tested c16 | |
| T4 | **Weyl for singular values.** For `A, E` of equal size, `|σᵢ(A+E) − σᵢ(A)| ≤ ‖E‖₂` for every `i`; and `‖E‖₂ ≤ ‖E‖_F`. | §3.3 | Horn & Johnson Cor. 7.3.5 (secondary only); Golub & Van Loan Cor. 8.6.2 (secondary only); Weyl 1912 attribution **unconfirmed** | proof below; tested c14 (exact, 50 digits) | |
| T5 | **TF32 matmul, rigorous form.** If both operands are rounded to nearest at 11 significant bits (`u_t = 2⁻¹¹`) before the product and the accumulation is fp32 in any order, then `|error| ≤ (2u_t + u_t²) Σ|aᵢbᵢ| + γₙ (1+u_t)² Σ|aᵢbᵢ|`. **The hypothesis "round to nearest" is H4, not a theorem.** | §3.4 | NVIDIA TF32 blog / Wikipedia for the 10-bit mantissa; the formula is mine | proof below; code uses a slightly smaller factor, see §3.4 | |
| T6 | **Forward error from backward error.** If a solver returns `x̂` that solves `(A+ΔA)x̂ = b` with `‖ΔA‖ ≤ c·u·‖A‖`, then `‖x̂−x‖/‖x‖ ≤ κ c u/(1 − κ c u)` (for `κ c u < 1`). | standard perturbation theory | Higham §1.6, ch. 7 (unconfirmed) | **Not used in the bound.** Statement stands as a conditional; `c` is an H-class fact for LU. c15 only shows the observed constant is below 0.5 | |

### Hardware and library facts (class H)

| id | statement | how it is checked | status |
|---|---|---|---|
| H1 | `+ − × ÷ √` are correctly rounded (so T0 applies per operation) on the device, in fp32 and fp64. | c01: 1e6 fp32 operands vs fp64 then rounded; 2e4 fp64 operands vs exact rationals. Standard reference IEEE 754-2019 §5 (clause unconfirmed; secondary only) | tested on 2080 Ti and H100. Not tested: subnormal inputs and flush-to-zero modes (H8) |
| H2 | FMA (if used) is a single correctly rounded `a·b+c`. T1 holds with or without it. | Not tested separately. Argument: FMA has at most one rounding, so the factor count in §3.1 does not increase | argued only |
| H3 | fp32 `exp, erf, log, tanh` error is at most about 2 ulp on CUDA (`exp` 1.87, `tanh` 1.78, `erf` 1.01, `log` 0.84 observed) and 0.8 ulp on CPU. So the rounding multiplier `k` for them is `k ≥ 2.5·max_ulp` in the code, not 1. Not a guaranteed bound; the CUDA programming guide lists maximum ulp errors but they are measured, not proven. | c02 (2e6 random points per function); c18 calibrates `k` per backend | tested on a sample, so the observed maximum is a lower bound on the true worst case. **Weak: calibrated on a sample of random points only** |
| H4 | TF32 is **off** by default for PyTorch matmul since 1.12 (`allow_tf32=False`), **on** by default for cuDNN, and **on** by default in ONNX Runtime on Ampere or newer (`use_tf32=1`). TF32 operand conversion rounds to nearest. | c08 (TF32 on/off error), c18 (TF32 variants); the rounding mode is **not directly tested**. PyTorch docs checked; ORT default stated in the onnx_error_bound README from experiments | defaults tested; rounding mode untested, see §4 |
| H5 | Matmul and `sum` are bit-reproducible for fixed shapes and stream; ops using float atomics (`index_add_`) are not. | c09 (100 repeats of 3 ops) | one size per op; not generalised |
| H6 | The computed SVD of `A` is the exact SVD of `A + E` with `‖E‖₂ ≤ p(m,n)·ε·‖A‖₂`, `p` a modest polynomial. **`p` is not specified by the source.** For d=16 we take `p ≤ 16`. | c13: exact zeros come out at ≤ 0.58 ε σ_max; prescribed `σ_min` sweep max error 4.25 ε σ_max (fp64), 2.6 (fp32). LAPACK Users' Guide (checked, `lug_old/node92`) | tested on rank-deficient integer and Hadamard matrices only; headroom 4× over the largest observation. **Weak: an a posteriori check (§4) is better than any constant** |
| H7 | fp64 `exp` and `erf` are accurate to a few ulp on the device. Needed for the float64 Jacobian (GELU′). | **Not tested.** c02 is fp32 only | **gap** |
| H8 | No flush-to-zero or subnormal-handling effects on the values we use. | **Not tested.** | **gap** |
| H9 | A cuBLAS/ORT matmul may use any summation order (blocked, split-K). T1 already covers any order, so no assumption is added. | follows from §3.1 | argued |

### Rules of thumb (class R, not for the bound)

| id | statement | what we know |
|---|---|---|
| R1 | Typical error of a sum or dot product is about `√n · u · Σ|xᵢyᵢ|`. | **Contradicted for GPU kernels** (c04: median is 4e-5 to 2e-4 of it at n=65536; c16: pairwise sums have flat error). The rigorous version, `√(n log n)·u` with high probability, is Higham & Mary (2019, checked) and needs independent, zero-mean rounding errors, an assumption, not a guarantee. |
| R2 | `‖σ_min error‖ ≈ d·ε·σ_max`, and numerical rank threshold `σ_max·max(M,N)·ε` (NumPy default). | A threshold convention, documented by NumPy (checked). It is H6 with `p=d`, nothing more. |
| R3 | Finite-difference best step `√u` / `u^(1/3)`. | c10 agrees. We use analytic Jacobians (match autograd to 1e-15), so this is not used. |
| R4 | The minimum of `σ_min` over N sample points approximates the infimum over the manifold. | Mine, unchecked, `L` unmeasured. An upper bound only. |

## 3. Proofs of the theorems

### 3.0 T0: round-to-nearest

Let `x` be in the normal range and `fl(x)` the nearest floating-point number (ties arbitrary), with spacing `s` between `fl(x)`'s neighbours at the binade of `x`. Then `|fl(x) − x| ≤ s/2`. The binade of `x` is `[2ᵉ, 2ᵉ⁺¹)`, with `s = 2ᵉ·2⁻ᵖ⁺¹` (`p` significand bits, so `u = 2⁻ᵖ`) and `|x| ≥ 2ᵉ`, so `|fl(x) − x| ≤ 2ᵉ·2⁻ᵖ = u·2ᵉ ≤ u|x|`. For the second form, if `|fl(x)| ≥ 2ᵉ` the same argument with `|fl(x)|` works. The only case with `|fl(x)| < 2ᵉ` is `fl(x) = 2ᵉ` reached from below, where `x` is in the binade `[2ᵉ⁻¹, 2ᵉ)`, the spacing below `2ᵉ` is `s/2`, and the error is at most `s/4 = u·2ᵉ/2 ≤ u·|fl(x)|`. So `|fl(x) − x| ≤ u·|fl(x)|`. (Hypothesis: normal range, so no subnormal spacing.) ∎

### 3.1 T1 and T3: the `γₙ` bound

**Lemma.** If `|δᵢ| ≤ u` and `nu < 1`, then `∏ᵢ₌₁ⁿ (1+δᵢ) = 1 + θₙ` with `|θₙ| ≤ γₙ`.
*Proof.* By induction. `n=1`: `|θ₁| ≤ u ≤ γ₁`. Step: `(1+θₖ)(1+δ) = 1 + θₖ + δ + θₖδ`, so `|θₖ₊₁| ≤ γₖ + u + γₖu = γₖ(1+u) + u`. With `γₖ = ku/(1−ku)`: `γₖ(1+u) + u = [ku(1+u) + u(1−ku)]/(1−ku) = (k+1)u/(1−ku)`. Since `1 − ku > 1 − (k+1)u > 0`, this is at most `(k+1)u/(1−(k+1)u) = γₖ₊₁`. ∎

**Dot product.** Computing `s = Σ xᵢyᵢ` in any order with a binary summation tree: each product is computed as `fl(xᵢyᵢ) = xᵢyᵢ(1+δ)` (H1), and each addition contributes a factor `(1+δ)` to every term below it in the tree. The tree on `n` leaves has depth at most `n−1`, so term `i` gets at most `1 + (n−1) = n` factors, and `fl(s) = Σ xᵢyᵢ(1+θₙ⁽ⁱ⁾)` with `|θₙ⁽ⁱ⁾| ≤ γₙ`. Hence `|fl(s) − s| ≤ γₙ Σ|xᵢyᵢ|`. With FMA, the number of roundings per term does not increase. ∎ For a plain sum (T3) there is no multiplication, so the depth bound is `n−1` and the bound is `γ_{n−1} Σ|xᵢ|`. ∎

(The bound is attained up to a constant: c16 sums `[2²⁴, 1, …, 1]` and reaches 0.9999 of it.)

### 3.2 T2: propagation rules

Let `x̂` approximate `x` with `|x̂ − x| ≤ eₓ`. Let `r̂ = fl(op(â, b̂))` be the computed result and `r = op(a, b)`. Write `ρ = |r̂ − op(â,b̂)|` for the rounding of this operation, which by T0 is at most `u|r̂|` when the operation is correctly rounded (`k=1`), and at most `k·u·|r̂|` for an operation with `k` times that error (H3). Then `|r̂ − r| ≤ |op(â,b̂) − op(a,b)| + ρ`. The first term is the propagated error:

| op | rigorous propagated term | proof sketch |
|---|---|---|
| `a ± b` | `eₐ + e_b` | triangle inequality |
| `a·b` | `|a|e_b + |b|eₐ + eₐe_b` | `âb̂ − ab = a(b̂−b) + b(â−a) + (â−a)(b̂−b)` |
| `a/b`, `e_b < |b|` | `(eₐ|b| + |a|e_b) / (|b|(|b|−e_b))` | `â/b̂ − a/b = ((â−a)b − a(b̂−b)) / (b̂b)` and `|b̂| ≥ |b| − e_b` |
| `√a` | `eₐ / (√a + √max(a−eₐ,0))` | `√â − √a = (â−a)/(√â+√a)` and `â ≥ a−eₐ` |
| `erf(a)` | `(2/√π)·exp(−max(|a|−eₐ,0)²)·eₐ` | mean value theorem on `[a−eₐ, a+eₐ]`, `|erf′|` is largest at the point closest to 0 |
| `Σ aᵢ` (n terms) | `Σ eᵢ + γ_{n−1} Σ(|aᵢ| + eᵢ)` | T3 applied to the perturbed inputs `âᵢ`, `|âᵢ| ≤ |aᵢ|+eᵢ` |
| `A·B` entry | `Σ|aᵢ|e_{bᵢ} + Σ eₐᵢ|bᵢ| + Σ eₐᵢe_{bᵢ} + γₙ Σ(|aᵢ|+eₐᵢ)(|bᵢ|+e_{bᵢ})` | the product rule per term plus T1 applied to `â, b̂` |

**Where the code is not the rigorous form** (found while writing this table, to be fixed or marked):

1. `chains.trace` for `sum` and for fp32 `matmul` uses `γₙ·Σ|a|·|b|` of the **true** values instead of `Σ(|a|+e)(|b|+e)`. The missing part is `γₙ` times the propagated error, which is below `1e-7·e`. The effect is negligible (second order) but the rule as written is not a proof. The TF32 rule already uses `|a|+e`.
2. `c03_propagation` tests `div` and `exp` with **first-order** rules (`ea/|b| + |a|eb/b²`, `exp·ea/10`), not the rigorous ones above. They passed (div max ratio 0.998) but "holds" for `div` there is an empirical statement about first-order accuracy, not a check of T2. The rigorous `div` rule is the one in c18.
3. `exp`'s own rounding is not correctly rounded (H3), so the c03 `exp` row is *expected* to exceed the bound. The measured 18% outside and 10.6× maximum are the evidence for H3, not a failure of T2.

### 3.3 T4: Weyl for singular values

Courant–Fischer for singular values: `σᵢ(A) = max over subspaces S with dim S = i of min over unit x in S of ‖Ax‖`. (This is Courant–Fischer for the eigenvalues of `AᵀA`, since `σᵢ² = λᵢ(AᵀA)` and `‖Ax‖² = xᵀAᵀAx`.)
For any unit `x`, `‖(A+E)x‖ ≥ ‖Ax‖ − ‖Ex‖ ≥ ‖Ax‖ − ‖E‖₂`. Take the minimum over `x ∈ S`, then the maximum over `S`: `σᵢ(A+E) ≥ σᵢ(A) − ‖E‖₂`. Swap `A` and `A+E` (replace `E` by `−E`) to get `σᵢ(A) ≥ σᵢ(A+E) − ‖E‖₂`. ∎
`‖E‖₂ ≤ ‖E‖_F` since `‖E‖₂² = λ_max(EᵀE) ≤ trace(EᵀE) = ‖E‖_F²`. ∎
Tightness: for `E = δ·qqᵀ` with `q` the right singular vector of `σ_min`, equality (c14: ratio exactly 1.0 in 50-digit arithmetic).

### 3.4 T5: TF32 matmul

Hypothesis (H4): each operand is replaced by `â = a(1+δ_a)`, `b̂ = b(1+δ_b)`, `|δ| ≤ u_t = 2⁻¹¹` (round to nearest at 11 significant bits). Then `|âb̂ − ab| = |ab|·|δ_a + δ_b + δ_aδ_b| ≤ (2u_t + u_t²)|ab|`. The tensor core multiplies `â·b̂` exactly (the product of two 11-bit numbers has 22 bits, representable in fp32 whose significand has 24) and accumulates in fp32, so T1 applies to the products of `â, b̂`: error ≤ `γₙ Σ|âb̂| ≤ γₙ(1+u_t)² Σ|ab|`. Total: `(2u_t + u_t² + γₙ(1+u_t)²) Σ|ab|`. The code (`2u_t + u_t² + γₙ`) omits the `(1+u_t)²` factor: a relative difference of about `4e-3·γₙ`, immaterial but not exact. If TF32 conversion **truncates** rather than rounds, each `δ` can be up to `2u_t` and the first term becomes about `4u_t`. c18's measured ratio of 0.05 to 0.10 of this bound would not distinguish the two cases, so the rounding mode needs its own test (§4). ∎

## 4. Tests still needed to close the weak rows

Each is small and is a falsification attempt, not a confirmation.

1. **H7 (fp64 erf/exp).** Same as c02 in fp64 against 50-digit mpmath on 1e5 points; report the maximum ulp.
2. **H4 (TF32 rounding mode).** Feed operands with an exactly known 12th significand bit (e.g. `1 + 2⁻¹¹ + 2⁻¹²`, `1 + 2⁻¹¹ + 2⁻¹³`) into a TF32 matmul with one nonzero term and read off the product. Round-to-nearest and truncation give different results.
3. **H6 (SVD error), replacing the constant by a certificate.** For the critical points only (DW.a point 0, DW.b worst points, a handful of random ones), recompute `σ_min(J)` with `mpmath.svd_r` on the exact float64 matrix. Then `|σ_min computed − σ_min exact|` is measured with no assumption about `p(m,n)`. This also gives the exact `σ_min` of the stored `J`.
4. **`E_J` (the Jacobian formula).** At the same points, compute `J` in mpmath from the same float64 weights and compare `‖J_float64 − J_mp‖₂` to the running-error bound for the formula (T2). This is the missing link between the forward-pass checks (c17, c18) and `σ_min`. Passing means the measured error is below the bound; the ratio says how loose it is.
5. **H3 over a wider range.** c02 draws uniformly from a range per function. Extend to the ranges the model sees (GELU argument magnitude, LayerNorm inputs).
6. **H8.** One check that subnormal inputs and outputs behave as the model needs (or that none occur in the Jacobian computation).

## 5. Pinned stack

Claims in class H hold only for the stack they were measured on. Pin and record in each `results.json`: GPU model and driver, CUDA version, `torch` (checked once: 2.13.0+cu130), `onnxruntime` (1.20.0, the custom `onnxruntime-training` build), `numpy`, `scipy`, `mpmath`. The checks currently record only the GPU name. Recording the versions is an easy change and makes a "holds" claim reproducible.

## 6. What this register does not claim

- It does not verify any literature source. Every `source unseen` / `unconfirmed` locator needs someone with the book to check the theorem number and statement. The proofs above make the *content* independent of those locators; the locators are only for credit.
- A claim in class H that passes its test is "not falsified on this stack by these inputs". Maxima over random samples are lower bounds on the true worst case.
- The proofs are mine and unreviewed. A colleague should read §3 and sign the "checked by" column before the guarantee is called verified.
