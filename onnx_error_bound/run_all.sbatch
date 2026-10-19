#!/bin/bash
#SBATCH --job-name=onnx-bound
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_%x.out

# Self test, calibration of three ONNX Runtime backends (CPU, CUDA fp32, CUDA TF32) on this node, export of the repo's models, validation of the bounds. From the repo root:
#   sbatch onnx_error_bound/run_all.sbatch
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD"
ORT_NV=/nfs-share/pa511/code_bases/onnxruntime_build_output/1.20.0_py311_cuda12_sm86/build_venv/lib/python3.11/site-packages/nvidia  # CUDA libraries the onnxruntime-training wheel was built against
ORTLIB="/usr/local/cuda-12.8/lib64:$ORT_NV/cudnn/lib:$ORT_NV/cublas/lib"
uv run python -m onnx_error_bound.selftest
for cfg in "ort_cpu cpu 0" "ort_cuda_fp32_h100 cuda 0" "ort_cuda_tf32_h100 cuda 1"; do
    set -- $cfg
    LD_LIBRARY_PATH="$ORTLIB" uv run python -m onnx_error_bound.calibrate --name $1 --provider $2 --tf32 $3
done
uv run python -m onnx_error_bound.export_repo_models
V=onnx_error_bound/validation
LD_LIBRARY_PATH="$ORTLIB" uv run python -m onnx_error_bound.validate --profiles ort_cpu ort_cuda_fp32_h100 ort_cuda_tf32_h100 \
    --models $V/models/trained.onnx $V/models/dw_a_point.onnx $V/models/dw_b_fold.onnx --inputs sphere=$V/inputs_sphere.npy normal=$V/inputs_normal.npy
