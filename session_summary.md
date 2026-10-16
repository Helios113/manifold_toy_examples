# Session summary: tangent pushforward, restricted Jacobian, injectivity, UMAP

Run analysed throughout: `runs_single/sphere_i2_s3_d16` (sphere, intrinsic dim 2, 3 splits / 9 classes, d_model 16, 5 checkpoints at
steps 1, 17, 283, 4757, 80000; final eval accuracy 0.993). "Space 0/1/2" = the input of resblock 0, the input of resblock 1, the output of
resblock 1. "Map 0/1" = resblock 0 / resblock 1. All jobs ran with `sbatch` on mauao. The HTML figures were not opened in a browser by me; I
checked scene counts, colour-bar limits and job logs only.

## 1. `block_inputs` is correct

`block_inputs` (`diagnostics.py`) pushes the manifold's analytic tangent basis through the (affine) embedding and then through each earlier
block's Jacobian, giving the tangent vectors `T` and the activations `h` at each block input. `test_all.py` (9 tests, all pass) covers it,
including `test_pushforward_matches_autograd` (jvp) and the analytic block Jacobian against `vmap(jacrev)`. A later cross-check on the trained
model gave `max |J_analytic - J_autograd| ~ 1e-15`.

## 2. Plots made (all from the trained run unless noted)

| Output | Script | What it shows |
|---|---|---|
| `plots/tangent_by_ckpt/step_*.pdf` | `visualise_tangent.py` | sphere and the tangent plane at `p*` through the three spaces, one PCA fit on the resblock 0 input, per checkpoint |
| `plots/angle_by_ckpt/step_*.pdf` | `visualise_angle.py` | angle between the weakest singular direction and the tangent plane, on the sphere, per checkpoint |
| `plots/geometry_by_ckpt/step_*/{01_kernel,02_alignment,03_orientation}.pdf` | `visualise_geometry.py` | kernel / alignment / orientation sphere maps per checkpoint |
| `plots/trained/06_training_curve.pdf` | `plot_training_curve.py` | train / eval loss and eval accuracy vs step, checkpoints marked |
| `plots/interactive/step_*/{tangent,kernel,alignment,angle,orientation,svd_sign}.html` | `visualise_interactive.py` | plotly versions (camera synced across panels; colour bars span the real data range), two columns (one per map) except `tangent.html` (three spaces) |
| `plots/injectivity/injectivity.html` | `check_injectivity.py` | `s_T` and `s_stack` on the sphere for engineered / fold / trained |
| `plots/umap/umap_layers.html` | `umap_layers.py` | 3-D UMAP of each space, engineered / fold / trained |

`engineered.py` holds `build_engineered` (the model of `visualise_synthetic.py`) and `build_fold` (below). `plot_training_curve.sbatch`,
`visualise_*.sbatch`, `check_injectivity.sbatch` and `umap_layers.sbatch` run them. `plotly` and `umap-learn` were added as dependencies.

## 3. What the diagnostics are (and are not)

Details in `restricted_jacobian.md`. In short:

- The restricted Jacobian `J|_M` is `J` applied to the tangent basis, the d x m matrix `JQ` (what `block_inputs` pushes forward).
- `R = Q^T J Q` (used in `diagnostics.restricted_jacobian`) is not `J|_M`: it projects the image back onto the **input** plane, so
  `det R = vol * overlap` and its sign flips with the overlap, not with the map degenerating.
- With a basis of the output plane, `S = Q'^T J Q` has `|det S| = vol` and the same singular values as `JQ`. Its sign is a convention
  (QR gives +vol, an SVD gives the arbitrary sign of `det V`; plotted in `svd_sign.html`: ~50% positive, no trend, so noise).
- The pushforward orientation is +1 at every point by definition; it only requires `vol > 0` (asserted for both maps at every checkpoint).
- A rank drop of the d x m matrix `JQ` has codimension d - m + 1 = 15, so no signed determinant can detect it. The detector is
  `sigma_min(J|_T)` (`s_T`), and `s_stack` (algebraic test that `ker J` meets the tangent plane).
- Ambient `det J` is a real sign and it does change: resblock 0 is negative at 19% of the sphere already at step 1 and 44% at step 80000,
  because the LayerNorm std of the input is only ~0.2 so the MLP branch is not a small perturbation of the identity. The analytic
  Jacobian agrees with autograd, so this is not a bug. The kernel direction is not in the tangent plane (see below).
- The formulas with an explicit target chart `h`, an implicit target `G = 0`, or the three-space chain (`J_{u->t}`, `J_{p->t}`) are all
  equivalent to `S` in some basis; with the target taken to be `F(g(M))` they reduce to the identity. A real signed test needs an
  independent orientation of the output, which we do not have.

## 4. Results

On the trained model the restricted map never degenerates: min `s_T` is 0.82 (map 0) and 0.42 (map 1), min `s_stack` 0.36 and 0.33, and
`vol > 0` everywhere, at every checkpoint. Meanwhile ambient `sigma_min(J)` is tiny (5e-6, 5e-5), so the kernel direction exists but stays out
of the tangent plane. The angle between the weakest singular direction and the tangent plane has medians 78.6 to 81.5 deg (resblock 0 input),
58.6 to 73.3 deg (resblock 1 input) and 78.4 to 89.2 deg (resblock 1 output), never close to 0.

Models compared in `check_injectivity.py` / `umap_layers.py` (6000 points):

| model | min `s_T`, map 0 | min `s_stack`, map 0 | output collisions (frac with far/nn < 2), map 0 | UMAP trustworthiness at space 1 |
|---|---|---|---|---|
| engineered | 0 (one point) | 3.5e-16 (one point) | 0.013 | 0.998 |
| fold | 3.5e-3 | 2.3e-3 | 0.465 | 0.835 |
| trained (step 80000) | 0.82 | 0.36 | 0 | 0.999 |

- **Engineered** (random init + a rank-1 edit of block 0's `W2`): `J(p*) v = 0` for one tangent vector at one point (`|J v| = 4.7e-16`). Both
  tests fire at exactly that point and nowhere else. It is a single-point, first-order degeneracy, so it loses no information (UMAP is
  unchanged), which is why it cannot be seen there.
- **Fold** (block 0 fitted so its output is the projection of the sphere along the top principal axis, i.e. the sphere onto a disk, 2-to-1;
  relative fit error 3.3e-3): `s_T` and `s_stack` drop to ~3e-3 along a curve, about 46% of points have a far-away sphere point mapped close
  to them, and UMAP trustworthiness falls from ~1 to 0.835 right after block 0. This is a real many-to-one map.
- **Trained**: UMAP trustworthiness 1.000 / 0.999 / 0.996 across the three spaces, i.e. the sphere survives all layers. (Trustworthiness
  only penalises false neighbours, not lost ones; continuity was not computed.)

## 5. Open items and caveats

- `diagnostics.py` (`det`, `overlap`, `frac_reversed`), the training-time logging in `train.py` and the PDFs in `plots/angle_by_ckpt/`
  and `plots/geometry_by_ckpt/` still use `R` and/or the "resblock 1 output" column that was dropped from the interactive figures.
- Planned but not built: plot `sigma_min(J|_T)` on the sphere per map in place of the `det R` / `log10 vol` row; a local refinement
  that minimises `s_T` from the sampled minimum.
- The fold check "`J T` against `(I - a a^T) T`" was not run. The logged `median ||J - (I - a a^T)||` (1.15) compares the full 16 x 16
  Jacobian and is dominated by the 13 normal directions the fit never constrained; ignore it.
- `plots/trained/05_tangent_flow.png` and the earlier combined `angle_vs_tangent.png` were deleted as replaced.
