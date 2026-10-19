#!/bin/bash
#SBATCH --job-name=noise-h100
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_%x.out

# The whole noise-floor suite on the H100 (ruapehu), into results/noise_floor_h100 so the RTX 2080 Ti results are kept. From the repo root:  sbatch analysis/noise_floor/run_h100.sbatch
cd "$SLURM_SUBMIT_DIR"
export NOISE_FLOOR_OUT=results/noise_floor_h100
for c in c01_rounding c02_transcendental c03_propagation c04_dot_bound c08_tf32 c09_nondeterminism c10_finite_diff c13_svd_floor c14_weyl_exact c15_cond_rule c16_sum_tight c17_running_bound; do
    echo "=== $c"
    uv run python -m analysis.noise_floor.$c || echo "FAILED $c"
done
uv run python -m analysis.noise_floor.plots
uv run python -m analysis.noise_floor.summary
