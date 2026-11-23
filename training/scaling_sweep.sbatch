#!/bin/bash
#SBATCH --job-name=manifold-scaling
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a_scaling.out

cd "$SLURM_SUBMIT_DIR"
# shared trackio dir on NFS so the dashboard sees cluster runs from the login node
export TRACKIO_DIR="$SLURM_SUBMIT_DIR/trackio_logs"
mkdir -p "$TRACKIO_DIR"
# TRACKIO_PROJECT=manifold-scaling (default) or override: sbatch --export=ALL,TRACKIO_PROJECT=foo training/scaling_sweep.sbatch
uv run python -m training.scaling_sweep
