#!/bin/bash
#SBATCH --job-name=manifold-scaling-continue
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a_scaling_continue_80.out

cd "$SLURM_SUBMIT_DIR"
export TARGET_TPP=80
uv run python -m training.scaling_continue
