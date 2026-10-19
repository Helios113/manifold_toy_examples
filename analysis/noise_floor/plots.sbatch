#!/bin/bash
#SBATCH --job-name=noise-plots
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_%x.out

# Redraw the noise-floor figures from the saved results (no recomputation). From the repo root:  sbatch analysis/noise_floor/plots.sbatch
cd "$SLURM_SUBMIT_DIR"
for d in results/noise_floor results/noise_floor_h100; do NOISE_FLOOR_OUT=$d uv run python -m analysis.noise_floor.plots; done
