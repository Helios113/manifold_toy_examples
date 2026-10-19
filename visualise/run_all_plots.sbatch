#!/bin/bash
#SBATCH --job-name=plots-all
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_%x.out

# Everything in plots.md, from the repo root:  sbatch visualise/run_all_plots.sbatch
#   1. degenerate weights DW.a / DW.b -> weights/*.pt (skipped if both already exist; delete them to regenerate)
#   2. all plot series from the trained run (runs_single/sphere_i2_s3_d16, train it first with training/train_single.sbatch) and the DW
#      weights -> plots/series*/ as .html (interactive) and .png
# PNG export needs Chrome; it is installed in .chrome/ (uv run plotly_get_chrome -y --path "$PWD/.chrome", the dir must exist) because
# the default ~/.local install is node-local and not visible on the compute nodes.
cd "$SLURM_SUBMIT_DIR"
export BROWSER_PATH="$PWD/.chrome/chrome-linux64/chrome"
if [ ! -f weights/dw_a_point.pt ] || [ ! -f weights/dw_b_fold.pt ]; then
    uv run python -m training.make_degenerate
fi
for s in series0_data series1_pca series2_degenerate series3_kernel_cos series4_sigma_lipschitz series4a_degenerate series5_umap; do
    uv run python -m visualise."$s"
done
