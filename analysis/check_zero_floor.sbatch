#!/bin/bash
#SBATCH --job-name=zero-floor
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_%x.out

cd "$SLURM_SUBMIT_DIR"
uv run python -m analysis.check_zero_floor
