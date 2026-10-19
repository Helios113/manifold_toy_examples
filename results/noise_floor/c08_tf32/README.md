# c08 TF32: noise of tensor-core matmuls

## What we test
That a matmul with TF32 enabled has a much larger error than fp32, of the order of the TF32 unit roundoff `u_TF32 = 2^-11 ~ 4.9e-4`, so a quantity near 1e-4 computed this way is below the noise floor.

## How
- Run on ruapehu (H100): TF32 needs compute capability >= 8; the RTX 2080 Ti on ngongotaha does not have it.
- `(256 x n) @ (n x 256)` for n in {16, 256, 4096}, fp32 inputs, with `torch.backends.cuda.matmul.allow_tf32` False and True. It has to be a real matrix-matrix product: a first version used matrix-vector products and TF32 never engaged.
- Reference: fp64 product of the same fp32 inputs. Metric: max over entries of error / `sum|x_i y_i|`.

## What the literature says
- TF32 has a 10-bit mantissa (11 bits of precision with the implicit bit), the same as FP16, with FP32's exponent range, and accumulates in FP32 (NVIDIA blog on TensorFloat-32, confirmed). `u_TF32 = 2^-11` is our derivation from that width.
- PyTorch: `matmul.allow_tf32` defaults to False since 1.12; `cudnn.allow_tf32` defaults to True (PyTorch CUDA semantics docs, confirmed; the flags are being deprecated in favour of `fp32_precision`).
- Expected: with both inputs rounded to 11 bits each product has relative error up to about `2 u_TF32`, so the TF32 error should be at most about `2 u_TF32 + gamma_n` and roughly 10^3 times the fp32 error.

## How we use this later
Measured max error / `sum|x_i y_i|` for a `256 x n x 256` matmul on the H100:

| n | fp32 (TF32 off) | TF32 on | ratio |
|---|---|---|---|
| 16 | 1.3e-7 | 4.7e-4 | 3500 |
| 256 | 1.0e-7 | 1.4e-4 | 1400 |
| 4096 | 4.4e-8 | 3.4e-5 | 770 |

- **Any matmul with TF32 on has noise of about 5e-4 times the sum of absolute products at small n** (the guaranteed bound is `2 u_TF32 + gamma_n` = about 1e-3). A value around 1e-4 computed this way is noise.
- For every threshold or noise-floor study keep TF32 off: `torch.backends.cuda.matmul.allow_tf32 = False` (the default in current PyTorch), and remember `torch.backends.cudnn.allow_tf32` defaults to True, which affects convolutions.
- The RTX 2080 Ti (ngongotaha) has no TF32, so results there cannot be affected; results from ruapehu or mauao can.
