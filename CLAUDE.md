# manifold_toy_example

## Running code

Run anything that does real work (training, sweeps, plotting over many runs) on the cluster, with `sbatch` (batch jobs, long sweeps) or `srun` (interactive / short jobs). Do not run it directly on the login node.

Use `uv run` inside the job, e.g. `sbatch` script that calls `uv run python -m training.sweep` (run from the repo root; scripts live in `training/`, `visualise/`, `analysis/`, `tests/`, shared modules stay at the root).

## GPU

Run everything on the GPU nodes, not CPU-only. Request GPUs with `--gres=gpu:N` and the `turing` partition pinned to a node (`-w ngongotaha`; this cluster rejects multi-node partitions without `-w`) GPU jobs are capped at 8 CPUs per GPU. The sbatch scripts (`sweep.sbatch`, `run_all.sbatch`) already do this.

Code should use `cuda` when available. The models are tiny and launch-bound, so avoid per-step host syncs (`.item()`, `F.one_hot`) in the training loop, and share each GPU between several worker processes (`workers_per_gpu` in `config.py`; CUDA needs the `spawn` start method).

so not use srun, always sbatch so we have logs -- use mauao or ruapehu.


DO NOT GO BEYOND WHAT I ASK FOR