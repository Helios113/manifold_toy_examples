#!/bin/bash
#SBATCH --job-name=injectivity
#SBATCH --partition=ampere
#SBATCH -w mauao
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
uv run python -m analysis.check_injectivity
