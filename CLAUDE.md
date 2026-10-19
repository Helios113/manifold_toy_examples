# manifold_toy_example

## Running code

Run anything that does real work (training, sweeps, plotting over many runs) on the cluster, with `sbatch` (batch jobs, long sweeps) or `srun` (interactive / short jobs). Do not run it directly on the login node.

Use `uv run` inside the job, e.g. `sbatch` script that calls `uv run python -m training.sweep` (run from the repo root; scripts live in `training/`, `visualise/`, `analysis/`, `tests/`, shared modules stay at the root).

## GPU

Run everything on the GPU nodes, not CPU-only. Request GPUs with `--gres=gpu:N` and the `turing` partition pinned to a node (`-w ngongotaha`; this cluster rejects multi-node partitions without `-w`) GPU jobs are capped at 8 CPUs per GPU. The sbatch scripts (`sweep.sbatch`, `run_all.sbatch`) already do this.

Code should use `cuda` when available. The models are tiny and launch-bound, so avoid per-step host syncs (`.item()`, `F.one_hot`) in the training loop, and share each GPU between several worker processes (`workers_per_gpu` in `config.py`; CUDA needs the `spawn` start method).

so not use srun, always sbatch so we have logs -- use mauao or ruapehu.

### Nodes

Always check which nodes are free
Pick the node by this order: ngongotaha if the job fits, else a free one of mauao / ruapehu (check `sinfo -N -o "%N %P %T"` and `squeue`). Always pin with `-w`. 

| Node | `--partition` | `-w` | GPUs | Use when |
|---|---|---|---|---|
| ngongotaha | `turing` | `ngongotaha` | 7 x RTX 2080 Ti (11 GB) | Default: tiny models, sweeps, anything that fits in 11 GB. float64 is slow here. |
| mauao | `ampere` | `mauao` | 8 x A40 (48 GB) | Does not fit on ngongotaha (memory) or ngongotaha is busy. |
| ruapehu | `hopper` | `ruapehu` | 4 x H100 (2 + 1 + 1) | Needs the most memory / fast float64, or the others are busy. |

Header for any of them: `#SBATCH --partition=<partition>`, `#SBATCH -w <node>`, `#SBATCH --gres=gpu:N`, `#SBATCH --cpus-per-task=<8*N max>`, `#SBATCH --output=logs/%j_%x.out`.


DO NOT GO BEYOND WHAT I ASK FOR

Run  things in a non-blocking way; don't wait for batched jobs to finish
Always run on a free node, unless told otherwise, if a job can run on ngog, then run it there


for onnx jobs always run them on ruapehu or mauao -- whichever is free


Do not keep suggesting things -- decide what is sufficient and execute it -- no need for endless continuiations