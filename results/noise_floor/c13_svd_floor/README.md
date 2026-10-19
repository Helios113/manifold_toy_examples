# c13 SVD floor: where does a computed singular value become zero?

## What we test
That a computed singular value is wrong by at most about `p eps sigma_max` with a small constant (we assumed `d = 16`), so an exact zero is returned below that level and nothing smaller can be resolved with confidence.

## How
- (a) Exact zeros: 2000 random integer matrices `B C` with `B` 16 x r and `C` r x 16, entries in [-3, 3], r = 15 and 8. Entries are exactly representable, so 16 - r singular values are exactly 0. GPU `svdvals` in fp32 and fp64; we report the computed value in units of `eps sigma_max`.
- (b) Prescribed values: `A = H diag(s) H^T` with `H` the 16 x 16 Hadamard matrix / 4 (exactly orthogonal), `sigma_max = 1` and `sigma_min = 10^-j` for j = 0..18, three permutations each. The matrix is rounded to the working precision and its exact singular values come from a 50-digit mpmath SVD, so only the algorithm's error is measured.

## What the literature says
- The SVD is backward stable: the computed singular values are the exact ones of `A + E` with `||E||_2 <= p(m,n) eps ||A||_2`, `p` a modestly growing function (LAPACK Users' Guide, "Error bounds for the SVD", confirmed). With Weyl's inequality (c14) each computed value is off by at most `p eps sigma_max`.
- NumPy's `matrix_rank` uses the threshold `sigma_max max(M,N) eps` (NumPy docs, confirmed), the same form as our `d = 16`.
- Expected: exact zeros come out below `d eps sigma_max`; computed `sigma_min` stays within that band of the exact value.

## How we use this later
This is the check that gives the zero threshold for `sigma_min`. Measured on 16 x 16 matrices on the GPU:

| precision | exact zero returned as (max) | error of a computed `sigma_min` (max) | `d = 16` threshold used so far |
|---|---|---|---|
| fp32 | 1.40 `eps sigma_max` | 2.6 `eps sigma_max` | 16 `eps sigma_max` = 1.9e-6 `sigma_max` |
| fp64 | 1.05 `eps sigma_max` | 4.25 `eps sigma_max` | 16 `eps sigma_max` = 3.6e-15 `sigma_max` |

- **Recommended: call a singular value numerically zero if it is below `16 eps sigma_max`.** This is 3.6e-15 `sigma_max` in fp64 (about 1e-14 for `sigma_max` around 2 to 3) and 1.9e-6 `sigma_max` in fp32. It is about 4 times the largest error seen (4.25 eps), so it has some margin.
- Using the measured 4.25 `eps` instead would give a tighter threshold (9.4e-16 `sigma_max` in fp64). I would keep 16 because the measured matrices are structured (Hadamard) and 16 x 16 only; the constant `p` is said to grow slowly with dimension.
- Apply it to the plotted `sigma_min(J)`: in fp64 anything above about 1e-14 is not zero from rounding. But J itself is built from rounded operations: add the error of J (c04, c17) through Weyl (c14) for the full floor.
