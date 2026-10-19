#!/bin/bash
#SBATCH --job-name=noise-speed
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%j_%x.out

# Check 19: speed of TF32 on/off in torch and ONNX Runtime (needs an Ampere or newer GPU, so ruapehu). From the repo root:  sbatch analysis/noise_floor/run_speed.sbatch
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD"
export NOISE_FLOOR_OUT=results/noise_floor_h100
uv run python -m analysis.noise_floor.c19_torch
ORT_VENV=/nfs-share/pa511/code_bases/onnxruntime_build_output/1.20.0_py311_cuda12_sm86/build_venv/lib/python3.11/site-packages/nvidia
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$ORT_VENV/cudnn/lib:$ORT_VENV/cublas/lib" uv run python -m analysis.noise_floor.c19_onnx
uv run python -m analysis.noise_floor.c19_report
uv run python -m analysis.noise_floor.plots
uv run python -m analysis.noise_floor.summary
