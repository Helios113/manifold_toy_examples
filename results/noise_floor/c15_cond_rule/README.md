# c15 condition number: forward error of a linear solve

## What we test
The rule of thumb that a backward-stable solve has relative forward error of about `kappa u` (condition number times unit roundoff).

## How
- `A = H diag(s) H^T` with singular values `s` geometric from 1 down to `10^-k`, k = 0..8 (fp32) and k = 0..16 (fp64), four permutations each. `x*` is a vector of small integers and `b = A x*` rounded to the working precision.
- GPU `torch.linalg.solve` (LU with partial pivoting). Reference: the exact solution of the same rounded system from mpmath `lu_solve`, so only the algorithm's error is measured. `kappa` is taken from the exact singular values of the rounded matrix.
- Metric: `||x_hat - x|| / ||x||` against `kappa`.

## What the literature says
- Higham Sec. 1.6 (section number unconfirmed): forward error is about the condition number times the backward error, and backward error is about `u` for a stable algorithm. This is a rule of thumb, not a theorem with a fixed constant.
- Expected: points on or below the line `kappa u`, slope 1 on log-log axes, reaching relative error near 1 when `kappa` approaches `1/u`.

## How we use this later
Use this whenever a result comes from solving a linear system or inverting a matrix (for example a kernel vector, a pseudo-inverse, projecting onto a tangent space). Relative error of the answer is about `kappa u`; measured maximum was 0.47 `kappa u` (fp32 0.45, fp64 0.47), median 0.02 to 0.05.

| precision | kappa | expected relative error (using 0.47 kappa u) |
|---|---|---|
| fp32 | 1e3 | 2.8e-5 |
| fp32 | 1e6 | 2.8e-2 |
| fp32 | 1e7 | 0.28 (no reliable digit beyond the first) |
| fp64 | 1e10 | 5.2e-7 |
| fp64 | 1e14 | 5.2e-3 |

- For a matrix with `sigma_min / sigma_max = 1e-3` (as in the degenerate models), `kappa = 1e3`: an fp32 solve loses about 4 to 5 digits, an fp64 solve about 3 and still has about 13 left.
- The value 0.47 is the maximum we saw, not a bound. For decisions, use `kappa u` itself as the working estimate.
- When `kappa` approaches `1/u` (1.7e7 fp32, 9e15 fp64), treat the matrix as singular for solving purposes.
