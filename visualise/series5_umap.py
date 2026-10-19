"""Series 5: 3-D UMAP of the three spaces, coloured by the position on the original sphere, for the last checkpoint, DW.a and DW.b.
Writes plots/series5_umap/<model>.{html,png}."""

import umap

from visualise.common import *  # noqa: F401,F403

def umap_figure(model, title, stem):
    clouds = [cloud(umap.UMAP(n_components=3, n_neighbors=15, random_state=0).fit_transform(x), sphere_rgb) for x in spaces(model)]
    save(three_scenes(f"{title}: 3-D UMAP (colour = position on the original sphere)", clouds), stem)


last_step, last_model = load_checkpoints()[-1]
for name, model in {f"trained_step_{last_step:07d}": last_model, **load_degenerate()}.items():
    umap_figure(model, name, OUT / "series5_umap" / name)
