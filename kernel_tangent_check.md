# Does the block's kernel overlap the manifold's tangent space?

Code: `diagnostics.py` (`check_block`, `restricted_jacobian`, `model_diagnostics`), Jacobian in `model.py`, all tests in `test_all.py`.
During training set `track_geometry=N` in the config to log these at every eval step (N manifold points); `uv run diagnostics.py <run_dir>` prints them for every checkpoint of a finished run.

## The question

At a data point, the manifold has a tangent space: the directions you can move while staying on the manifold.
A residual block has a Jacobian J: how it responds to a small nudge of its input. Its kernel is the set of nudges it
completely ignores (J v = 0).

We ask: is there a direction that is both tangent to the manifold **and** ignored by the block?

## Setting it up

- The tangent space is known analytically for the sphere and the plane (`Manifold.tangent_basis`), but in the input space.
- The block sits inside the model, so we first push the tangent directions forward to where the block sees them:
  through the embedding, and through any earlier block (multiply by its Jacobian).
- J itself is computed in closed form (`ResMLPBlock.jacobian`) and agrees with autograd.

## The check

1. **Simplest version.** Take the tangent directions, apply J, and look at the smallest "stretch" (singular value) among
   them. If it is exactly zero, some tangent direction is ignored, so the answer is yes. Call this `s_T`.
2. **Using the kernel directly.** The kernel of this block can be written down: every kernel vector has the form
   `-W2 u`, where `u` solves a small linear system (the `(I + K) u = 0` system in the code). Asking that such a vector
   also lies in the tangent space is one more linear condition. Stack the two conditions into one matrix; its smallest
   singular value, `s_stack`, is zero exactly when the answer is yes. This is the more accurate version.

`check_block` returns, per point:

| name | what it tells you |
|---|---|
| `s_stack`, `s_T` | zero means the kernel and the tangent space share a direction; small means close to it |
| `s_J` | smallest stretch of J overall; if this is not zero there is no kernel at all |
| `kernel_dim` | how many dimensions the kernel has |
| `align` | how much the block's weakest direction points along the tangent space (1 = fully) |

## What to expect

A residual block is "identity plus something", so it only loses information if that something exactly cancels the
identity along some direction. That is rare, so for a normal block the kernel is empty and the answer is no. The numbers
worth looking at are then how small `s_T` gets compared with `s_J`, and `align`: how close the tangent space comes to the
block's weakest direction.

## Caveats

- `s_T` and `s_J` are only accurate down to about 1e-8 (float64). `s_stack` is accurate to about 1e-15, so use it, in
  float64, when deciding whether a small number is really zero.
- The pushed-forward tangent space must keep its full dimension; a very small `d_model` can already squash it.
- `test_all.py` builds a block whose kernel is known exactly and checks that the overlap is found when the
  tangent space contains it and not otherwise.

## The restricted Jacobian and the sign of its determinant

`restricted_jacobian(J, T)` in `diagnostics.py`. Restricting the block to the manifold, the Jacobian only acts on tangent
directions: that is `J Q`, a d-by-m matrix (Q = orthonormal tangent basis). A determinant needs a square matrix, so we
express the result back in the same tangent basis, `R = Q^T J Q` (m-by-m). Its determinant does not depend on which
tangent basis you pick, and it splits into two parts:

- **`vol`** (always >= 0): how much the block stretches small patches of the manifold.
- **`overlap`** (between -1 and 1): the signed cosine between the tangent space and its image after the block, both seen
  in the residual stream. Positive means the block keeps the orientation of the tangent space, negative means it reverses
  it (for a curve, the block turns the direction of travel around), and 0 means the image is rotated at right angles to
  some tangent direction.

`det R = vol * overlap`, so **the sign of the determinant is the sign of `overlap`**. It is also returned as `sign`.
`sign_det_J` is the sign of the full (unrestricted) Jacobian for comparison.

Notes:
- The sign is only meaningful per block (the tangent space at that block's input versus its image under that block); it
  does not chain across blocks.
- Because a residual block is "identity plus something", the sign is expected to be positive unless the block's
  change is large; in the test on a random block it is positive everywhere.
- `test_all.py` also checks the restricted Jacobian on hand-built cases (identity, flips, scalings, a 90-degree rotation), basis
  independence, and a random block. It has not been run on trained models.
