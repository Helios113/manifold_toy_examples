# c09 non-determinism: are identical GPU computations bit-identical?

## What we test
Which GPU operations give different results when run twice on the same input. Float addition is not associative, so any kernel whose summation order varies (for example one using atomics) is not reproducible.

## How
- 100 repetitions of the same call, comparing each result with the first: `A @ B` (2048 x 2048), `v.sum()` over 2*10^7 values, and `index_add_` of 2*10^7 values into 100 bins.
- Metric: max |run - first run| / max |first run|, and the number of runs that differ.

## What the literature says
- PyTorch's reproducibility notes say floating-point addition is not associative, so `atomicAdd` on floats can give different rounding on each evaluation; they list `index_add_`, `scatter_add_` and `bincount` among the affected operations (confirmed; matmul and `sum` are not on the list we saw).
- Reordering `m` additions changes the result by at most `gamma_m` relative to `sum|x_i|` and typically by about `sqrt(m) u` (Higham Ch. 3 and Higham & Mary 2019, see c04). Here `m = 2*10^5` additions per bin.
- Expected: `matmul` and `sum` bit-identical; `index_add_` different in most runs, with differences below `gamma_m`.

## How we use this later
- Matmul and a large `sum` were bit-identical over 100 runs on the RTX 2080 Ti. `index_add_` differed in 100 of 100 runs by up to 1.1e-5 of the result's scale.
- Rule: any quantity produced with float atomics (`index_add_`, `scatter_add_`, `bincount`, and the backward of pooling, `interpolate`, `embedding_bag`) has run-to-run noise of about **1e-5 of its scale** in fp32 (guaranteed bound `gamma_m`, about 1e-2 at m = 2*10^5). Two runs that differ by less than that are the same result.
- This noise is separate from rounding: it exists even if inputs and code are identical, so it also sets a floor for "did my change alter the number" comparisons.
- To remove it, use deterministic algorithms (`torch.use_deterministic_algorithms(True)`, which according to the PyTorch docs raises an error when no deterministic version exists). Other GPUs and library versions may differ, so rerun c09 on the target hardware.
