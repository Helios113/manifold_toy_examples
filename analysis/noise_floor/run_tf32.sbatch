#!/bin/bash
#SBATCH --job-name=noise-tf32
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_%x.out

# TF32 needs an Ampere+ GPU, so this one check runs on ruapehu (H100). From the repo root:  sbatch analysis/noise_floor/run_tf32.sbatch
cd "$SLURM_SUBMIT_DIR"
uv run python -m analysis.noise_floor.c08_tf32
uv run python -m analysis.noise_floor.plots
uv run python -m analysis.noise_floor.summary
