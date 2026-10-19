"""Series 1: points in the three spaces, one 3-PCA fit on the resblock 0 input, one figure per training checkpoint.
Writes plots/series1_pca_by_step/step_<step>.{html,png}."""


from visualise.common import *  # noqa: F401,F403

for step, model in load_checkpoints():
    pca_figure(model, f"Trained, step {step}", OUT / "series1_pca_by_step" / f"step_{step:07d}")
