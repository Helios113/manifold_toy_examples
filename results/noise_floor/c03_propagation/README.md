# c03 propagation: first-order error propagation through one operation

## What we test
That the computed error of `a+b`, `a*b`, `a/b` is bounded by the propagated input errors plus a fresh rounding error `u |result|`:
- add: `ea + eb + u|r|`
- mul: `|a| eb + |b| ea + ea eb + u|r|`
- div: `ea/|b| + |a| eb/b^2 + u|r|`
- `exp(a/10)`: `exp(a/10) ea/10 + u|r|` (this one is expected to be violated, see below).

## How
- 10^6 pairs of fp64 values with magnitudes in [1e-2, 1e2], random signs. They are rounded to fp32, so the input errors `ea = |fl32(a) - a|` and `eb` are known exactly.
- The operation is run in fp32 on the GPU and compared with the exact result computed in fp64 from the original `a, b`.
- Reported: error / bound per trial (CDF), and the fraction of trials within the bound.

## What the literature says
- These are the first-order rules of running error analysis (Higham Sec. 3.3, section number unconfirmed; originally Wilkinson, *Rounding Errors in Algebraic Processes*, 1963), built on the standard model of c01.
- Expected: 100% within the bound for `+ * /` because they are correctly rounded. `exp` is not correctly rounded (c02), so its true rounding error can exceed `u|r|` and some trials should fall outside the bound.

## How we use this later
These are the per-operation rules a tracer applies to every node of a graph (c17 does exactly this):

| operation | propagated error (absolute) | plus fresh rounding |
|---|---|---|
| `a + b` | `ea + eb` | `u |r|` |
| `a * b` | `|a| eb + |b| ea (+ ea eb)` | `u |r|` |
| `a / b` | `ea/|b| + |a| eb/b^2` | `u |r|` |
| `f(a)` smooth | `|f'(a)| ea` | `k_f u |r|` with `k_f` from c02 (1 only for correctly rounded ones) |

Measured here: for `+ * /` the actual error never exceeded the bound (max error / bound 0.998 to 0.99994, median about 0.3, so the bound is tight to a factor of about 3). For `exp` with `k = 1` only 82% of trials were inside and the worst was 10.6 times the bound, which is why `exp` needs the larger `k` from c02. To confirm a new `k`, rerun c03 with that factor.
