"""Series 4: smallest singular value of J (linear scale) on the original sphere, with the input vs output distance scatter and the histogram of the
lower Lipschitz values of each block underneath (see common.sigma_lipschitz_figure), per training checkpoint.
Writes plots/series4_sigma_lipschitz/step_<step>.{html,png}."""

from visualise.common import *  # noqa: F401,F403

for step, model in load_checkpoints():
    sigma_lipschitz_figure(f"Step {step}: sigma_min(J) and lower Lipschitz values", model, OUT / "series4_sigma_lipschitz" / f"step_{step:07d}")
