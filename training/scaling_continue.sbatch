#!/bin/bash
#SBATCH --job-name=manifold-scaling-continue
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a_scaling_continue.out

cd "$SLURM_SUBMIT_DIR"
# continue every finished run in runs_scaling to 40 tokens per param (override with TARGET_TPP=...)
uv run python -m training.scaling_continue
