#!/bin/bash
#SBATCH --job-name=export-onnx
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_%x.out

cd "$SLURM_SUBMIT_DIR"
uv run python -m training.export_onnx
