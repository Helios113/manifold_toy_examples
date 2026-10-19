# c10 finite differences: how noisy is a numerical derivative?

## What we test
That a finite-difference derivative has a best achievable error of about `sqrt(u)` (forward) or `u^(2/3)` (central), reached at a step `h ~ sqrt(u)` or `h ~ u^(1/3)`, far worse than an analytic derivative.

## How
- `f(x) = exp(x)`, whose derivative is exactly `exp(x)`, at 1000 points in [0.5, 1.5].
- 37 steps `h` from 1e-9 to 1, forward `(f(x+h) - f(x))/h` and central `(f(x+h) - f(x-h))/(2h)`, in fp32 and fp64 on the GPU. `h` is the nominal step, as a naive user would use it.
- Metric: median relative error over the points. Overlaid theory: `h/2 + u/h` (forward) and `h^2/6 + u/h` (central), for `exp` up to O(1) constants.

## What the literature says
- Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 8: truncation error grows with `h`, rounding error as `u/h`, so the optimum balances them. The exact statement could not be confirmed from a source I could read; the orders `sqrt(u)` and `u^(2/3)` are the standard step-size analysis.
- Expected best h and error: fp32 forward ~2.4e-4 / ~2.4e-4; fp32 central ~3.9e-3 / ~1.5e-5; fp64 forward ~1.1e-8 / ~1.1e-8; fp64 central ~4.8e-6 / ~2.3e-11.

## How we use this later
Best achievable accuracy of a finite-difference derivative (measured, median over 1000 points, `exp(x)`):

| precision | forward: best error at h | central: best error at h |
|---|---|---|
| fp32 | 1.8e-4 at 3e-4 | 6.1e-6 at 3e-3 |
| fp64 | 6.6e-9 at 1e-8 | 7.7e-12 at 3e-6 |

- If a Jacobian were computed by finite differences, each entry would carry a relative error at least this large, and by Weyl (c14) so would every singular value (relative to `sigma_max`). So **finite-difference `sigma_min` values below about 1e-11 (fp64) or 6e-6 (fp32) times `sigma_max` mean nothing.**
- The analytic Jacobian avoids this and is limited only by the rounding of c01 to c04 (about 1e-15 in fp64). In an earlier run its entries matched autograd to 1.3e-15 relative. So never use finite differences to decide whether something is zero; use them only as a cross-check with these tolerances.
- The best `h` is also fixed by the precision: about `sqrt(u)` (forward) and `u^(1/3)` (central), not "as small as possible".
