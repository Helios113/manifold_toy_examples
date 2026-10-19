#!/bin/bash
#SBATCH --job-name=train-single
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_%x.out

# One run: CONFIG in config.py (sphere, intrinsic dim 2, 3 splits, d_model 16), logging the kernel / tangent-intersection /
# restricted-Jacobian diagnostics at every eval. Output: runs_single/<name>/ (metrics.json has the per-eval "geometry").
cd "$SLURM_SUBMIT_DIR"
uv run python -m training.train
