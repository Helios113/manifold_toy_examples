# The restricted Jacobian, `det R`, and what detects a degenerate restricted map

Notation: `f` is a block (resblock 0 or 1), `J` its ambient Jacobian at a point (d x d, d = 16), `M` the sphere (m = 2),
`Q` (d x m) an orthonormal basis of the tangent plane `T_pM` in the block's input space. The tangent vectors are pushed
through the earlier blocks by `block_inputs` (`T_1 = J_0 T_0`, ...).

## 1. The restricted Jacobian `J|_M`

`J|_M` is the linear map `T_pM -> T_{f(p)} f(M)`. On the tangent basis it acts as the d x m matrix `JQ`: column `i` is the image
of the tangent vector `q_i`, and the image lies in the output tangent plane `span(JQ)`.

To write it as an m x m matrix, pick a basis on each side: input `Q`, output `Q'` (orthonormal basis of `span(JQ)`, from the QR
factorisation `JQ = Q' S`). Then

    S = Q'^T J Q          (m x m, the restricted Jacobian in those bases)
    |det S| = vol = sqrt(det((JQ)^T JQ)) = prod of singular values of JQ     (area stretch)
    (JQ)^T (JQ)  is the pullback metric.

## 2. What `R` is

The code (`diagnostics.restricted_jacobian`) uses

    R = Q^T J Q,   det R = vol * overlap,   overlap = det(Q^T Q') in [-1, 1].

`R` is `J|_M` followed by the orthogonal projection onto the **input** plane `T_pM`, not onto the output plane `J T`. So it is not
the restricted Jacobian. It identifies the input and output tangent planes through the residual stream (a block is `I + update`)
and asks how far the block rotated the plane. `overlap < 0` means the image plane is rotated past 90 degrees from the input
plane; it does not mean the map flips the manifold's orientation.

## 3. Orientation of the restricted map

The sign of `det S` depends only on how the output plane is oriented. With the **pushforward orientation** (the output plane is
oriented so that `(J q_1, J q_2)` is positive, given `(q_1, q_2)` positive on the sphere), the matrix of `J|_M` in those bases is
the identity, so the sign is +1 at every point **by definition**. It says nothing about the network. What it does need is
`vol > 0` (so `J|_M` is injective and the orientation is defined); then it is a continuous orientation of the image surface.

A non-trivial sign needs an orientation of the output that does not come from the map:

- ambient `det J` (orientation of the whole stream): real and basis-free, it does change sign where `J` is singular;
- the identity identification (`det R`): see section 2;
- an independent frame on the output surface (not canonical in 16-D): not tried.

## 4. Detecting that the restricted map becomes singular

The restricted map is singular where `JQ` loses rank, i.e. `ker J` meets `T_pM`. A signed determinant cannot detect this:

- A rank drop of a d x m matrix has codimension `d - m + 1` (here 16 - 2 + 1 = 15), so on a 2-D manifold it generically never
  happens, and there is no hypersurface of zeros for a sign to flip across.
- A determinant's zero set is a hypersurface (codimension 1), which is what makes a sign flip a detector. For `JQ` the only natural
  scalar is `vol >= 0`, which can touch zero but never changes sign.
- `det R` does flip, but it vanishes when the image plane is orthogonal to some input tangent direction (codimension 1), at points
  where `vol` is still well above zero. In the run it is negative on up to 35% of the sphere while `vol > 0` everywhere, so its
  flips are false alarms for this question.

The right detector is `sigma_min(J|_T)`, the smallest singular value of `JQ` (`s_T` in `check_block`): it is zero exactly when the
restricted map fails to be injective. It cannot guarantee detection the way a sign flip would (a narrow dip can fall between
samples), but the failure is codimension 15.

## 5. Numbers (run `runs_single/sphere_i2_s3_d16`, sphere, intrinsic dim 2, 3 splits, d_model 16)

From the 5 evals that logged geometry (256 points each):

| map | min `sigma_min(J\|_T)` | min `vol` |
|---|---|---|
| resblock 0 | 0.83 (step 283) | 1.19 |
| resblock 1 | 0.35 (step 4757) | 0.16 |

On 6000 points per checkpoint (resblock 0 / resblock 1):

| step | `det J < 0` | `det R < 0` |
|---|---|---|
| 1 | 19% / 0% | 0% / 0% |
| 17 | 33% / 0% | 0% / 0% |
| 283 | 53% / 34% | 33% / 11% |
| 4757 | 48% / 39% | 35% / 13% |
| 80000 | 44% / 21% | 28% / 6% |

`vol > 0` at every point, both maps, every checkpoint, so the restricted map never degenerates on these samples. The ambient
`det J` does change sign (`J` is singular somewhere on the sphere, `min sigma_min(J)` ~ 5e-6 at step 80000), but the kernel
direction is not in the tangent plane. The analytic `J` matches autograd to ~1e-15, so the sign changes are not a numerical artefact.
For resblock 0 the LayerNorm std of the input is only ~0.2 at init (factor 1/sigma ~ 5 in `J_ln`), so the MLP branch is not a small
perturbation of the identity (median `||J - I||` is 2.2 at step 1, 5.5 at step 80000).

## 6. Open

- Plot `sigma_min(J|_T)` on the sphere per map (real-range colour bar) in place of the `det R` / `log10 vol` row in
  `visualise_interactive.py`.
- `diagnostics.py` (`det`, `overlap`, `frac_reversed`), the training-time logging and the PDFs in `plots/geometry_by_ckpt/` still
  use `R`.
