# Synthetic-weight experiments

Code: `visualise_synthetic.py` (run with `sbatch visualise_synthetic.sbatch`), diagnostics from `diagnostics.py`.
Output: `plots/synthetic/`.

## Goal

See what the kernel, the ambient Jacobian and the restricted Jacobian of a residual block look like, without training.
We build a model, write weights into it by hand, and call the same evaluation functions that `train.py` logs during
training (`block_inputs`, `block_point_metrics`, `restricted_jacobian`). So the plots show exactly what the training
diagnostics would show for a block with that structure.

## Setting

- Data manifold: the sphere S^2, rotated into R^16 by a fixed random orthogonal matrix (`Manifold("sphere", 2, 16, 9, seed 0)`).
- Model: `ToyModel`, embed (16 -> 16), two pre-norm blocks `x + W2 gelu(W1 LN(x) + b1) + b2` with d_hidden = 4 * 16 = 64,
  final LayerNorm, unembed to 9 classes. Everything in float64, on CPU (`block_inputs` copies the model to CPU).
- Only block 1 is analysed. Its input is `embed(x)`, and the tangent space at each point is the manifold's analytic
  tangent basis pushed through the affine embedding (`embed.weight @ tangent_basis`).
- 3000 uniform points on the sphere. Point p* is the first of them.

## Quantities

At each point, for block 1 with input h and pushed-forward tangent basis T (16 x 2), Q = orth(T):

| name | meaning |
|---|---|
| J | ambient Jacobian df/dx of the block at h (16 x 16), closed form `I + W2 diag(gelu') W1 diag(gamma) J_ln` |
| sigma_min(J) | smallest singular value; 0 iff the kernel is non-empty |
| det J | sign = whether the block preserves orientation of the residual stream |
| s_stack | 0 iff ker J contains a tangent direction (exact test through the explicit kernel) |
| J Q | restricted Jacobian (16 x 2): J applied to the tangent space |
| R = Q^T J Q | restricted Jacobian written back in the tangent basis (2 x 2); det R = vol * overlap |
| |Q^T v_min| | how much the weakest right singular vector of J lies in the tangent space |

## How the weights are picked

All three variants start from the same default PyTorch initialisation (seed 0), so differences between them come only
from the edit.

1. **random init.** No edit. A residual block is identity plus a small perturbation, so J is close to I and the kernel is empty.
2. **W2 x alpha.** Multiply block-1's second linear layer W2 by alpha, trying 1, 2, 4, ... until det J < 0 at >= 20% of
   the sphere (alpha = 2 was enough). This makes the MLP branch strong enough to cancel the identity in some directions,
   which is what makes det J change sign.
3. **engineered kernel at p*.** Choose a unit tangent vector v = T[p*, :, 0]. Let
   `w = diag(gelu'(a)) W1 diag(gamma) J_ln v`, the vector so that `J v = v + W2 w`. A kernel vector needs `W2 w = -v`.
   The rank-1 edit
   `W2' = W2 + (-v - W2 w) w^T / |w|^2`
   gives `W2' w = -v` exactly, so `J(p*) v = 0` to machine precision (|J v| ~ 5e-16). Only W2 changes, so gelu', J_ln and
   the pre-activation at p* are unchanged. At other points the Jacobian changes by a rank-1 amount, so the kernel is
   exact at p* only.

## Current figures

- `01_sphere_maps.png`: for each variant, scatter on the sphere of log10 sigma_min(J), det J, log10 s_stack, det R.
- `02_point_detail.png`: at p*, for random and engineered: J as a matrix, singular values of J, weakest singular vector
  against the tangent basis, J Q, and R.

## Results so far (block 1)

- Random init: no kernel (sigma_min(J) ~ 2e-2 at p*), det J < 0 at 14% of points, det R > 0 everywhere.
- W2 x 2: det J < 0 at ~30%, det R still > 0 everywhere.
- Engineered: sigma_min(J(p*)) = 1.6e-16, the weakest direction is in the tangent plane (|Q^T v| = 1.000), s_stack ~ 1e-15
  at p* only, and det R < 0 at 7% of points near p*.

## Known limits

- The exact kernel is only created at one point. The sampled points almost never hit it, so "kernel at 0% of points"
  does not mean the engineered variant has no kernel.
- Only block 1 is studied; block 2 would need the tangent space pushed through block 1's Jacobian (`block_inputs` already does this).
- The engineered edit is not a trained solution; it shows what an exact overlap looks like, not whether training finds one.
