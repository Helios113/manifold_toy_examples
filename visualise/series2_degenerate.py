"""Series 2: the same 3-PCA plot for the degenerate weights DW.a and DW.b.
Writes plots/series2_degenerate/{dw_a_point,dw_b_fold}.{html,png}."""


from visualise.common import *  # noqa: F401,F403

for name, model in load_degenerate().items():
    pca_figure(model, name, OUT / "series2_degenerate" / name)
