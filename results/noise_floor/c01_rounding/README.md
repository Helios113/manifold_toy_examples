# c01 rounding: is every basic operation correctly rounded?

## What we test
That `+ - * / sqrt` on the GPU return the correctly rounded result, i.e. the relative error of one operation is at most the unit roundoff `u` (fp32: 2^-24, fp64: 2^-53).

## How
- fp32: 10^6 random pairs with magnitudes log-uniform in [1e-3, 1e3] and random signs, evaluated on the GPU. Reference: the same operation in fp64, rounded to fp32. This is exact for `+ - *` and, for `/` and `sqrt`, double rounding is harmless because 53 >= 2*24+2.
- fp64: 20000 pairs, checked against exact rational arithmetic (Python `Fraction`). `sqrt` is checked by testing that the input lies between the squares of the midpoints to the neighbouring floats.
- Reported: max relative error / u, and the fraction of results bit-identical to the correctly rounded value.

## What the literature says
- IEEE 754 requires `+ - * / sqrt` to be correctly rounded in the selected rounding mode (confirmed via secondary sources: arXiv 2404.00387, ISO/IEC 60559:2020 proposal P3864R1; the standard text itself was not seen).
- Hence the standard model `fl(a op b) = (a op b)(1 + d)`, `|d| <= u` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Ch. 2; the exact section number is unconfirmed).
- Expected: max error / u just below 1 (the bound is attained for inputs just above a power of two), and 100% bit-identical.

## How we use this later
The number to carry is the unit roundoff: **every `+ - * / sqrt` adds a relative error of at most `u`**: 6.0e-8 in fp32, 1.1e-16 in fp64 (measured max: 1.00 u in fp32 and 0.99 u in fp64 for sqrt).

What happens when operations are chained:
- `k` operations in a row, worst case: relative error `k u` (more exactly `gamma_k = k u / (1 - k u)`). Typical: about `sqrt(k) u`. For example 10 operations: at most 6e-7 in fp32, 1.1e-15 in fp64.
- **sqrt does not accumulate.** If the input has relative error `delta`, the output has `delta/2 + u`. Taking sqrt repeatedly converges to `2u` (1.2e-7 fp32, 2.2e-16 fp64), it does not grow.
- Worked example, the LayerNorm in this repo: `sigma = sqrt(mean(c^2) + eps)` with d = 16. The 16 squares, the sum of 16 positive terms and the `+ eps` give a relative error of at most `gamma_17` on the variance (the division by 16 is exact). The sqrt halves it and adds `u`: **`sigma` has relative error at most 5.7e-7 in fp32 and 1.05e-15 in fp64** (worst case; each further op, such as dividing `c` by `sigma`, adds at most `u`).
- In the error tracer (c17) this is the rule "result error `+= u |result|`" for these five operations.
