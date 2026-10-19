#!/bin/bash
#SBATCH --job-name=noise-floor
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j_%x.out

# All checks except TF32 on ngongotaha (RTX 2080 Ti, Turing: no TF32). From the repo root:  sbatch analysis/noise_floor/run.sbatch
cd "$SLURM_SUBMIT_DIR"
for c in ${@:-c01_rounding c02_transcendental c03_propagation c04_dot_bound c09_nondeterminism c10_finite_diff c13_svd_floor c14_weyl_exact c15_cond_rule c16_sum_tight c17_running_bound}; do
    echo "=== $c"
    uv run python -m analysis.noise_floor.$c || echo "FAILED $c"
done
uv run python -m analysis.noise_floor.plots
uv run python -m analysis.noise_floor.summary
