#!/bin/bash
#SBATCH --job-name=make-dw
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_%x.out

cd "$SLURM_SUBMIT_DIR"
uv run python -m training.make_degenerate
