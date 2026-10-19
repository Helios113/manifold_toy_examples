# c14 Weyl: how much can singular values move under a perturbation?

## What we test
Weyl's inequality for singular values, `max_i |sigma_i(A+E) - sigma_i(A)| <= ||E||_2`, including that it is attained, so the error in `sigma_min` can really be as large as the error in the matrix.

## How
- 50-digit arithmetic (mpmath), so rounding cannot affect the test. `A = H diag(s) H^T` with `s = 1 - i/20` (15 values) and `sigma_min = 1e-3`, so its singular values are exactly `s`.
- Worst case: `E = delta q q^T` with `q` the column of `H` belonging to `sigma_min`. This shifts `sigma_min` by exactly `delta`.
- Random `E` (3 per `delta`) scaled to `||E||_2 = delta`. `delta` runs from 1e-12 to 1e-2.
- Metric: `max_i |d sigma_i| / ||E||_2`.

## What the literature says
- Horn & Johnson, *Matrix Analysis* (1985), Cor. 7.3.5, and Golub & Van Loan, *Matrix Computations*, 4th ed., Cor. 8.6.2: `|sigma_i(A+E) - sigma_i(A)| <= ||E||_2` (both confirmed only through papers that cite them). It follows from Weyl's eigenvalue inequality applied to the symmetric matrix `[[0, A], [A^T, 0]]`.
- Expected: ratio at most 1 for all `E`, exactly 1 for the worst-case `E`, and below 1 for random `E`.

## How we use this later
This is the rule that turns an error in a matrix into an error in its singular values: **the error in every singular value, `sigma_min` included, is at most `||E||_2`, the spectral norm of the error in the matrix.** It is attained, so it cannot be tightened in general.

Recipe for the noise floor of `sigma_min(J)` of a computed Jacobian:
1. Bound the error of each entry of J with the tracer (c17) or c04, giving an error matrix `E` of absolute bounds.
2. `||E||_2 <= ||E||_F = sqrt(sum E_ij^2)`; this is the noise floor of any singular value of J.
3. Add the SVD's own error, about 1 to 4 `eps sigma_max` (c13).

Example with made-up but typical sizes: 256 entries each with error 1e-7 (fp32) gives `||E||_F` = 1.6e-6. So `sigma_min` values in fp32 below about 2e-6 are noise, close to the `16 eps sigma_max` of c13 for `sigma_max` of order 1.
