# c04 dot bound: error of inner products and matmul

## What we test
That `|fl(x.y) - x.y| <= gamma_n sum|x_i y_i|` with `gamma_n = n u / (1 - n u)`; that the typical error is much smaller (about `sqrt(n) u sum|x_i y_i|` or less); and that the error is controlled relative to `sum|x_i y_i|`, not relative to the result, so it is huge relative to the result under cancellation.

## How
- n in {16, 64, 256, 1024, 4096, 16384, 65536}, 100 trials each, fp32 on the GPU (TF32 off), reference in fp64 from the same fp32 inputs.
- Three input kinds: normal, positive (no cancellation), and cancel (`y` projected so that `x.y = 0` exactly in fp64).
- Two kernels: elementwise product then `torch.sum`, and cuBLAS `bmm`.
- Reported: max and median error / `sum|x_i y_i|`, and median error / |result| for the cancel case.

## What the literature says
- The deterministic bound `gamma_n |x|^T |y|` (Higham Ch. 3; the form with `nu/(1-nu)` is confirmed in Ipsen's slides, the lemma number is unconfirmed).
- Under independent zero-mean rounding errors, `gamma_n` can be replaced by about `sqrt(n log n) u` with high probability (Higham & Mary, SIAM J. Sci. Comput. 41(5), 2019; confirmed). This is a model, not a guarantee.
- Expected: all points below `gamma_n`; typical error well below `sqrt(n) u`; error / |result| of order 1 or larger when the result is 0.

## How we use this later
For an fp32 inner product or matmul entry over `n` terms, the guaranteed error is `gamma_n sum|x_i y_i|`, but that is very pessimistic. Measured max error / `sum|x_i y_i|` (fp32, GPU, all n from 16 to 65536):

| input | measured max | in units of u |
|---|---|---|
| normal | <= 9.4e-8 | <= 1.6 u |
| positive | <= 3.9e-7 | <= 6.6 u |
| cancelling (`x.y = 0`) | <= 4.7e-8 | <= 0.8 u |

| n | guaranteed `gamma_n` (fp32) |
|---|---|
| 16 | 9.5e-7 |
| 64 | 3.8e-6 |
| 1024 | 6.1e-5 |
| 65536 | 3.9e-3 |

- Rule for the project: the absolute error of an entry of a matmul is about `<= 4e-7 * sum|x_i y_i|` in practice (guarantee: `gamma_n`). It scales with the sum of absolute products, **not with the entry itself**.
- That is why cancellation hurts: when the exact result is 0, the error relative to the result was 1 to 4 (100% to 400%) at every n tested. A Jacobian entry that results from cancelling terms of size 1 has absolute noise of about 1e-7, however small the entry is.
- In the layers here the inner dimension is 16 (embedding, blocks) or 64 (MLP hidden), so the guaranteed `gamma_n` is 9.5e-7 to 3.8e-6 in fp32 and 1.8e-15 to 7e-15 in fp64.
