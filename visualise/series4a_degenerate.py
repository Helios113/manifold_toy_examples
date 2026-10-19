"""Series 4A: the same as series 4 for the degenerate weights DW.a and DW.b.
Writes plots/series4a_degenerate/{dw_a_point,dw_b_fold}.{html,png}."""

from visualise.common import *  # noqa: F401,F403

for name, model in load_degenerate().items():
    sigma_lipschitz_figure(f"{name}: sigma_min(J) and lower Lipschitz values", model, OUT / "series4a_degenerate" / name)
