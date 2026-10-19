# c02 transcendental: are exp / erf / log / tanh correctly rounded?

## What we test
That library implementations of transcendental functions are accurate to a few ulp but NOT correctly rounded (error can exceed 0.5 ulp), so the "`u |result|`" rounding term of the standard model underestimates their error.

## How
- 2*10^6 random fp32 inputs per function: `exp` in [-20, 20], `erf` in [-4, 4], `log` in [1e-3, 1e3], `tanh` in [-5, 5].
- fp32 result on the CPU and on the GPU, against the same function evaluated in fp64 (whose own error, ~1e-16, is negligible next to an fp32 ulp).
- Error measured in ulp of the fp32 result; we plot the tail (fraction of inputs with error > x ulp) and the maximum.

## What the literature says
- Correct rounding is required by IEEE 754 only for `+ - * / sqrt` (see c01); for transcendental functions libraries document a bound of a few ulp.
- The CUDA C++ Programming Guide has a table of maximum ulp errors for single-precision functions; NVIDIA states these come from extensive but not exhaustive testing, so they are not guaranteed bounds (seen via NVIDIA's mirror; the `expf`/`erff` rows were not seen). A forum report found an observed error above the documented bound for `erfcinvf`.
- Expected: CPU and GPU maxima above 0.5 ulp and at most a few ulp, with GPU values typically larger than CPU.

## How we use this later
For `exp`, `erf`, `log`, `tanh` do **not** use the single-operation `u |result|` of c01. Use a multiple of it. Measured maximum error over 2*10^6 inputs, fp32 (1 ulp is at most `2u` of the value, so relative error <= `2 * ulp * u`):

| function | CPU max (ulp) | GPU max (ulp) | GPU relative error bound to use |
|---|---|---|---|
| exp | 0.74 | 1.87 | `3.7 u` = 2.2e-7 |
| erf | 0.69 | 1.01 | `2.0 u` = 1.2e-7 |
| log | 0.89 | 0.84 | `1.7 u` = 1.0e-7 |
| tanh | 0.51 | 1.78 | `3.6 u` = 2.1e-7 |

- Where this matters here: the analytic Jacobian uses `erf` and `exp` (the exact GELU and its derivative). In fp32 on the GPU each call contributes up to about 2e-7 relative error, more than ten times the 6e-8 of a plain add.
- These are maxima over a random sample, not guarantees. They were measured in fp32 only; fp64 `exp`/`erf` were not tested, so don't reuse these factors there.
- In the tracer (c17), give these functions the rounding term `k u |result|` with k from the last column's coefficient (about 4 for exp/tanh on the GPU, 2 for erf).
