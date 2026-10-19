# c17 running bound: error tracing through a computation

## What we test
That a small tracer carrying `(value, absolute error bound)` through `+` and `*` gives a valid bound for every instance of a longer computation, and how pessimistic it is. This is the method we would apply to the graph of a larger model.

## How
- Tracer rules: add: `ea + eb + u|r|`; mul: `|a| eb + |b| ea + ea eb + u|r|`; inputs are exact fp32 values with zero error.
- Horner evaluation of random polynomials (degree 4, 16, 64, 256; coefficients and `x` uniform in [-1, 1]) and dot products (n = 64, 1024; normal entries), 5000 instances each, fp32 on the GPU.
- Reference: the same computation in fp64 from the same fp32 inputs. Reported: the fraction of instances with actual error <= traced bound, and median bound / actual.

## What the literature says
- This is running error analysis (Higham Sec. 3.3, section number unconfirmed; Wilkinson 1963), using the standard model of c01 and the propagation rules of c03.
- Expected: the bound holds for every instance (it is a rigorous bound under the rounding model) but is pessimistic, since errors partly cancel; the probabilistic analysis of Higham & Mary (2019) explains why typical errors are well below worst-case bounds.

## How we use this later
This is the tool for a noise-floor number for a specific computation. Write the computation with the `Tr` class (value + error bound) from `c17_running_bound.py`, with `k_f u |result|` from c02 for transcendental functions, and read off the error bound of the output. For the Jacobian, trace it to get an error matrix, then apply Weyl (c14) and the SVD error (c13) to get the floor of `sigma_min`.

How much to trust the output (measured, fp32):

| computation | bound holds | bound / actual error (median) |
|---|---|---|
| Horner, degree 4 | 100% | 6.1 |
| Horner, degree 16 to 256 | 100% | 8 |
| dot product, n = 64 | 100% | 26 |
| dot product, n = 1024 | 100% | 99 |

- The traced bound is an upper limit: use it as "numbers below this are noise", not as the expected error. Expect it to overstate by 10 to 100 times for long sums (the typical error grows like `sqrt(n)`, the bound like `n`; Higham & Mary 2019).
- The empirical alternative (rerun in fp64 and compare, or perturb inputs, as in the methods list of noise_floor.md) gives the typical error; the tracer gives the guarantee. Reporting both brackets the true floor.
- Still to do: the tracer has only `+` and `*`; division, sqrt and the c02 functions need their rules added before applying it to the LayerNorm and GELU of the model.
