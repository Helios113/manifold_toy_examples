#!/bin/bash
#SBATCH --job-name=noise-chains
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%j_%x.out

# Check 18: chains of operators in torch and in ONNX Runtime. The custom onnxruntime-training wheel is built for sm86 (A40/mauao): its CUDA provider cannot run on the Turing
# RTX 2080 Ti (ngongotaha, those variants are recorded as failed) and can only run on the H100 (ruapehu, sm90) if the embedded PTX is JIT-compiled. From the repo root:
#   sbatch analysis/noise_floor/run_chains.sbatch                                   (ngongotaha: torch + ORT CPU)
#   sbatch analysis/noise_floor/run_chains_h100.sbatch                              (ruapehu: also ORT CUDA)
cd "$SLURM_SUBMIT_DIR"
export NOISE_FLOOR_OUT=results/noise_floor_h100
export PYTHONPATH="$PWD"
uv run python -m analysis.noise_floor.c18_torch
ORT_VENV=/nfs-share/pa511/code_bases/onnxruntime_build_output/1.20.0_py311_cuda12_sm86/build_venv/lib/python3.11/site-packages/nvidia  # cuDNN / cuBLAS the onnxruntime-training wheel was built against; set for this step only, torch brings its own CUDA libraries
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$ORT_VENV/cudnn/lib:$ORT_VENV/cublas/lib" uv run python -m analysis.noise_floor.c18_onnx
uv run python -m analysis.noise_floor.c18_analyse
uv run python -m analysis.noise_floor.plots
uv run python -m analysis.noise_floor.summary
