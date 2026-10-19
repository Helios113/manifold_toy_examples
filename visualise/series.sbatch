#!/bin/bash
#SBATCH --job-name=series
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_%x.out

# One plot series (or several), from the repo root:  sbatch visualise/series.sbatch series3_kernel_cos series4_sigma_lipschitz
# Scripts: series0_data  series1_pca  series2_degenerate  series3_kernel_cos  series4_sigma_lipschitz  series4a_degenerate  series5_umap  series_scaling_training
# Needs runs_single/sphere_i2_s3_d16 (training/train_single.sbatch) and, for series 2, 4A and 5, weights/ (sbatch training/make_degenerate.sbatch).
cd "$SLURM_SUBMIT_DIR"
export BROWSER_PATH="$PWD/.chrome/chrome-linux64/chrome"  # plotly PNG export; see run_all_plots.sbatch
for s in "$@"; do
    uv run python -m visualise."$s"
done
