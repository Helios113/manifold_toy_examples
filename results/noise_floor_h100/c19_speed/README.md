# c19 speed: how much faster is TF32?

Figure: `plot.png`. Run on the H100 (`results/noise_floor_h100/c19_speed/`); raw timings in `raw_torch.json` and `raw_onnx.json`.

## What we test
How much faster fp32 matmuls and a whole residual MLP block become with TF32 tensor-core math on (torch `allow_tf32=True`, ONNX Runtime CUDA `use_tf32=1`) than with it off, and at which sizes it pays.

## How
- Square `N x N` matmul for N = 16 ... 8192, and the block chain (LayerNorm, matmul, bias, GELU, matmul, bias, residual) at three sizes (batch, width, hidden): S = (4096, 16, 64) is this repo's model, M = (4096, 256, 1024), L = (16384, 1024, 4096).
- torch: CUDA events, 20 warm-up calls, median of 5 loops; the block is run op by op (eager). ONNX Runtime: IO binding with inputs on the GPU (no host copies), all graph optimisations on, `perf_counter` around the call (it returns after the stream finishes), median of 5 loops.
- Speedup = time with TF32 off / time with TF32 on.

## What the literature says
TF32 runs on tensor cores (Ampere and newer), with 10 mantissa bits and FP32 range, accumulating in FP32 (NVIDIA TensorFloat-32 blog, confirmed earlier). Tensor cores only pay when a matmul is large enough to be compute-bound; tiny kernels are limited by launch and memory time. I did not look up the H100's published peak numbers, so none are quoted here.

## What we found
| | N = 16 | 64 | 256 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| matmul speedup, torch | 1.26 | 1.25 | 1.31 | 4.08 | 4.99 | 4.70 | 5.94 |
| matmul speedup, ONNX Runtime | 1.08 | 1.13 | 1.22 | 2.80 | 4.48 | 5.58 | 7.05 |

Throughput at N = 8192: 36 TFLOP/s fp32, 216 (torch) and 257 (ONNX Runtime) TFLOP/s with TF32.

| whole block | S (16 / 64) | M (256 / 1024) | L (1024 / 4096) |
|---|---|---|---|
| torch, off -> on (ms) | 0.112 -> 0.105 | 0.328 -> 0.211 | 10.26 -> 3.76 |
| torch speedup | 1.07 | 1.56 | 2.73 |
| ONNX Runtime, off -> on (ms) | 0.079 -> 0.084 | 0.328 -> 0.218 | 10.35 -> 3.68 |
| ONNX Runtime speedup | 0.94 | 1.50 | 2.81 |

## How we use this later
- **For this repo's model (width 16, hidden 64): no gain from TF32** (1.07 and 0.94 for the whole block; the block takes 0.08 to 0.11 ms either way, so it is launch-bound). Since TF32 costs about 1000 times more error per matmul (c08, c18), keep it off for the model and for every noise-floor study.
- It pays from about width 1000 upward: matmuls of N >= 1024 run 3 to 7 times faster; whole blocks 1.5 times at width 256 and 2.7 to 2.8 times at width 1024.
- If TF32 is used for speed on a large model, predict its error with the TF32 rule of c18 (`2 u_tf32 + u_tf32^2 + gamma_n` per matmul).
- Caveat: one GPU (H100 PCIe), one library version each, timings of the whole block include non-matmul kernels. ONNX Runtime's S result (0.94) is within what I would expect from run-to-run timing noise; I did not measure the spread.
