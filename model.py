import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    """Pre-norm residual block: x + MLP(LayerNorm(x))"""

    def __init__(self, d_model: int, d_hidden: int | None = None):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))

    @torch.no_grad()
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic Jacobian df/dx of the block at x, (B, d) -> (B, d, d), J[b, i, j] = d f_i / d x_j.

        f(x) = x + W2 gelu(W1 LN(x) + b1) + b2, so
            J = I + W2 diag(gelu'(a)) W1 diag(gamma) J_ln,   a = W1 LN(x) + b1,
        with gelu'(a) = Phi(a) + a phi(a) (exact erf GELU, the nn.GELU default) and, for LN(x) = gamma * xhat + beta with
        xhat = (x - mean) / sigma, sigma = sqrt(var + eps),
            J_ln = (I - 1 1^T / d - xhat xhat^T / d) / sigma.
        """
        d = x.shape[-1]
        lin1, act, lin2 = self.mlp
        assert isinstance(act, nn.GELU) and act.approximate == "none", "analytic gelu' below is for the exact GELU"
        c = x - x.mean(-1, keepdim=True)
        sigma = (c.pow(2).mean(-1, keepdim=True) + self.norm.eps).sqrt()  # (B, 1)
        xhat = c / sigma
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        J_ln = (eye - 1.0 / d - xhat[:, :, None] * xhat[:, None, :] / d) / sigma[:, :, None]  # (B, d, d)
        a = lin1(self.norm(x))  # pre-activation, (B, d_hidden)
        gelu_grad = 0.5 * (1 + torch.erf(a / 2**0.5)) + a * torch.exp(-0.5 * a * a) / (2 * torch.pi) ** 0.5  # Phi(a) + a phi(a)
        inner = lin1.weight[None] * 1.0  # (1, d_hidden, d)
        J_mlp = lin2.weight @ (gelu_grad[:, :, None] * lin1.weight @ (self.norm.weight[:, None] * J_ln))  # (B, d, d)
        return eye + J_mlp


class ToyModel(nn.Module):
    """Linear embed (intrinsic_dim -> d_model), two pre-norm x+MLP blocks, final LayerNorm, unembed (d_model -> n_classes)."""

    def __init__(self, intrinsic_dim: int, d_model: int, n_classes: int, d_hidden: int | None = None):
        super().__init__()
        self.embed = nn.Linear(intrinsic_dim, d_model)
        self.blocks = nn.Sequential(
            ResMLPBlock(d_model, d_hidden),
            ResMLPBlock(d_model, d_hidden),
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.unembed(self.final_norm(self.blocks(self.embed(x))))  # logits
