# Numerical noise floor: facts used and references

Status of each reference: **checked** = the statement or citation was confirmed against a web source (named in the item); **checked (secondary)** = confirmed only through a paper/page that cites the original, not the original itself; **unconfirmed** = I searched and could not find the source text, so the statement is a standard result from memory and the section/theorem number must be verified in the book before quoting; **mine** = my own argument, not from the literature.

Notation: `u` = unit roundoff (round to nearest). `eps` = machine epsilon = `2u` (what `numpy.finfo(...).eps` returns).

| dtype | `u` | `eps` |
|---|---|---|
| float32 | 2^-24 ≈ 6.0e-8 | 2^-23 ≈ 1.2e-7 |
| float64 | 2^-53 ≈ 1.1e-16 | 2^-52 ≈ 2.2e-16 |

## 1. Rounding model

1. **Every basic operation is correctly rounded**: `fl(a op b) = (a op b)(1 + δ)`, `|δ| ≤ u`, for `op` in `+ - * /` and `sqrt`.
   - IEEE 754-2019 / ISO/IEC 60559:2020: addition, subtraction, multiplication, division and square root must be correctly rounded in the selected rounding mode. **Checked (secondary)**: <https://arxiv.org/pdf/2404.00387> and the C++ proposal <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3864r1>; the standard's own text was not seen. The clause numbers are unconfirmed.
   - Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Chapter 2 "the standard model". The chapter is **unconfirmed** by the table of contents (the search returned only the front matter of the book); the section number 2.2 is unconfirmed.
2. **Transcendental functions (`exp`, `erf`, `log`, ...) are not guaranteed correctly rounded**; their error is a few ulp and depends on the library/GPU implementation. No single constant applies. Check the vendor's documented ulp error: the CUDA C++ Programming Guide has a table of maximum ulp errors for single-precision functions, generated from extensive but not exhaustive testing, so they are not guaranteed bounds (seen via NVIDIA's mirror, <https://developer.nvidia.com/zh-cn/blog/cuda-math-method-cn>; **checked (secondary)**). I did not see the `expf`/`erff` rows. A forum report found observed error above the documented bound for `erfcinvf` (<https://forums.developer.nvidia.com/t/larger-than-expected-documented-error-in-erfcinvf/111299>), so verify against your CUDA version.

## 2. Error of one operation and of sums / dot products

3. **Add/subtract/multiply, first order.** If the inputs `a, b` carry errors `ea, eb`:
   - `a ± b`: `err ≈ ea + eb + u|result|`
   - `a * b`: `err ≈ |a| eb + |b| ea + u|result|`
   - general smooth `f`: `err ≈ |f'(a)| ea + u|result|`
   These follow from item 1 plus a first-order Taylor expansion. This is the rule set used by *running error analysis*: Higham Section 3.3 (**unconfirmed**, section number not seen); originally Wilkinson, *Rounding Errors in Algebraic Processes*, Prentice-Hall / HMSO 1963 (reprinted by SIAM). The book and its date are **checked** (<https://nhigham.com/2023/06/20/wilkinsons-rounding-errors-book-reprinted-by-siam/>); that it contains running error analysis is **unconfirmed**.
4. **Dot product / matmul entry with n terms**: `|fl(xᵀy) − xᵀy| ≤ γ_n |x|ᵀ|y|`, with `γ_n = n u / (1 − n u)` (valid for `n u < 1`).
   - Higham Chapter 3. `γ_n = nu/(1−nu)` **checked** via the FLAME notes (<https://www.cs.utexas.edu/~flame/laff/alaff/chapter06-backward-error-analysis-of-dot-general-case.html>). The inner-product bound with `|x|ᵀ|y|` and `nu/(1−nu)`, attributed to Higham (2002), is **checked (secondary)** in Ipsen's slides (<https://ipsen.math.ncsu.edu/ps/slides_SIAM_CSE2019.pdf>). The "Lemma 3.1" and "Section 3.1" numbers are **unconfirmed**.
   - The bound is relative to `|x|ᵀ|y|`, the sum of the absolute values of the products, **not** to the result. This is why cancellation gives large relative error.
5. **Probabilistic version**: if the rounding errors are independent, zero-mean random variables, `γ_n` can be replaced by roughly `√(n log n) · u`, with high probability (the common "replace n by √n" rule of thumb is the informal version).
   - Higham & Mary, "A New Approach to Probabilistic Rounding Error Analysis", *SIAM J. Sci. Comput.* 41(5), A2815–A2835, 2019. **Checked** (citation, the `√(n log n)` form, and the independence assumption): <https://eprints.maths.manchester.ac.uk/2673/>.
   - Caveat from the same literature: the independence assumption is a model and may not hold, so this is not a guaranteed bound.
6. **Forward error ≈ condition number × u** (rule of thumb for a backward-stable algorithm): Higham Section 1.6 (**unconfirmed**; section number not seen).

## 3. Perturbation of singular values (Weyl)

7. **Weyl's inequality for singular values**: for two matrices `A, B` of the same size, `|σ_i(A) − σ_i(B)| ≤ ‖A − B‖₂` for every `i`. So the error in `σ_min(J)` is at most the spectral norm of the error in `J`. Since `‖·‖₂ ≤ ‖·‖_F`, a Frobenius-norm bound on the error matrix also works.
   - Horn & Johnson, *Matrix Analysis*, 1985, Corollary 7.3.5. **Checked** against a secondary source that cites exactly this: <https://arxiv.org/pdf/2602.03682> (Theorem A.4 there); the book itself was not seen. The 2nd ed. may number it differently.
   - Original (eigenvalue form): H. Weyl, "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen (mit einer Anwendung auf die Theorie der Hohlraumstrahlung)", *Math. Ann.* 71, 441–479, 1912, DOI 10.1007/BF01456804. The paper's existence and citation are **checked** (<https://zenodo.org/record/1526112>), but that search described it only as the source of the Weyl *law* for PDE eigenvalue counting. I could not confirm that the matrix eigenvalue inequality is in it, so treat this attribution as **unconfirmed**.
   - It follows from the Hermitian eigenvalue form via the augmented matrix `[[0, A],[Aᵀ, 0]]`, whose eigenvalues are `±σ_i(A)`: <https://en.wikipedia.org/wiki/Weyl%27s_inequality> (**checked**).
   - Also: Golub & Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins 2013, Corollary 8.6.2, stated as `|σ_j(M+E) − σ_j(M)| ≤ ‖E‖₂`. **Checked (secondary)**: cited in exactly this form by <https://arxiv.org/pdf/1906.04327>; the book was not seen.
8. **A computed SVD is backward stable**: the computed singular values are the exact singular values of `A + E` with `‖E‖₂ ≤ p(m,n) u ‖A‖₂`, `p` a low-degree polynomial in the dimensions. Together with item 7, the SVD step adds an error of order `p(m,n) u σ_max`, which is why a threshold of the form `σ_max · n · eps` is used.
   - **Checked**: the LAPACK Users' Guide, "Error Bounds for the Singular Value Decomposition": <https://www.netlib.org/lapack/lug_old/node92.html>. It states the computed SVD is the exact SVD of `A + E` with `‖E‖₂ ≤ p(m,n) ε ‖A‖₂`, `p(m,n)` a modestly growing function, so each computed singular value is off by at most about `p(m,n) ε ‖A‖₂`. Small singular values therefore get no high relative accuracy.
   - Golub & Van Loan Section 8.6 and Demmel, *Applied Numerical Linear Algebra*, SIAM 1997, Chapter 5: **unconfirmed** (book text not seen; Demmel's Math 221 notes only say the SVD is covered in Chapter 5, <https://people.eecs.berkeley.edu/~demmel/ma221_Fall24/Lectures/Lecture_03.pdf>).
9. **The standard numerical-rank threshold**: `tol = σ_max · max(M, N) · eps`. This is what `numpy.linalg.matrix_rank` uses by default, and it is documented as the MATLAB algorithm. **Checked**: <https://numpy.org/doc/2.1/reference/generated/numpy.linalg.matrix_rank.html>.

## 4. Hardware and tooling

10. **TF32 has a 10-bit mantissa** (same as FP16, with FP32's 8-bit exponent), used only inside Tensor Core multiplies, with accumulation in FP32. With the implicit bit that is 11 bits of precision, so `u = 2^-11 ≈ 4.9e-4` for the products (the 2^-11 value is my derivation from the mantissa width). **Checked** (mantissa and range): <https://blogs.nvidia.com/blog/2020/03/24/tensorfloat-32-precision-format> and <https://en.wikipedia.org/wiki/TensorFloat-32>. PyTorch controls it with `torch.backends.cuda.matmul.allow_tf32` (default **False** since PyTorch 1.12) and `torch.backends.cudnn.allow_tf32` (default **True**, so convolutions may use TF32 even when matmuls don't). **Checked**: <https://docs.pytorch.org/docs/stable/notes/cuda>. The current docs also add an `fp32_precision` setting and say the `allow_tf32` flags will be deprecated, so check your installed version.
11. **GPU reductions can be order-nondeterministic** (atomics, different kernel/split choices), so the same fp32 input can give different results across runs. **Checked**: the PyTorch docs say floating-point addition is not associative, so `atomicAdd` on floats can give different rounding on each evaluation (<https://docs.pytorch.org/docs/2.3/notes/randomness.html>). The page names specific ops (`index_add_`, `scatter_add_`, `bincount`, pooling/interpolate backward, ...); I did not verify that matmul/reduction kernels in this repo are affected.
12. **Finite-difference derivatives have error ~ √u (forward) or u^(2/3) (central)** at the best step size, so a finite-difference Jacobian is far noisier than the analytic one. Nocedal & Wright, *Numerical Optimization*, 2nd ed., Chapter 8 (finite-difference derivative approximation). The statement is **unconfirmed**: searches found only course notes listing the chapter, not the text. The `√u` and `u^(2/3)` orders are the standard step-size analysis (forward step `~√u`, central step `~u^(1/3)`).

## 5. Empirical noise-floor methods

13. **Compare precisions**: run the same computation in a higher precision (fp64) and take the difference. General practice; I have no specific reference for it.
14. **Stochastic / Monte Carlo arithmetic** (random rounding or random input perturbation, then look at the spread):
    - D. S. Parker, *Monte Carlo Arithmetic: exploiting randomness in floating-point arithmetic*, 1997. The work and year are **checked** (WorldCat record <https://search.worldcat.org/title/Monte-Carlo-arithmetic-:-exploiting-randomness-in-floating-point-arithmetic/oclc/37892696>, and cited as the primary reference by <https://arxiv.org/pdf/1509.01347>). The report number "UCLA CSD-970002" is **unconfirmed**.
    - Verrou: F. Févotte, B. Lathuilière, "VERROU: a CESTAC evaluation without recompilation", SCAN 2016. **Unconfirmed**: the search returned nothing for this title, so the title and venue are from memory and may be wrong. Look up Verrou's own documentation for the correct citation.
    - CADNA: F. Jézéquel, J.-M. Chesneaux, "CADNA: a library for estimating round-off error propagation", *Comput. Phys. Commun.* 178(12), 933–955, 2008, DOI 10.1016/j.cpc.2008.02.003. **Checked**: <https://hal.archives-ouvertes.fr/hal-01146486>.
15. **Interval arithmetic** gives rigorous but usually pessimistic enclosures: R. E. Moore, *Interval Analysis*, Prentice-Hall, Englewood Cliffs NJ, 1966. **Checked**: <https://catalogue.i2m.univ-amu.fr/bib/9856>, <https://arxiv.org/pdf/physics/0302034>.

## 6. Not from the literature (my own reasoning)

16. **Sampling-resolution argument.** A minimum of `σ_min` over N sample points only upper-bounds the true infimum over the manifold. If the singular set is a point or curve and `σ_min` changes at rate `L` per unit distance, then the nearest sample can be up to `h` away (`h ≈ √(4π/N) ≈ 0.046` for N = 6000 points on S²), so a sampled minimum of order `L·h` is consistent with a true zero. **Mine; not checked, and L has not been measured for this model.**
17. **Applying items 3-9 to this repo's Jacobian** (`ResMLPBlock.jacobian`, [model.py](model.py)): the plots compute J in float64 (`block_inputs` in [diagnostics.py](diagnostics.py) casts to `.double()`), so the rounding floor for `σ_min` is about `σ_max · 16 · 2.2e-16 ≈ 1e-14`. The measured values for the DW models are in the log of job 61786 (`logs/61786_zero-floor.out`). The bound `σ_max · d · eps` is a heuristic of item 9's form, not a rigorous bound for this particular J.
