import torch
import torch.nn as nn


def _gelu_grad(act: nn.GELU, a: torch.Tensor) -> torch.Tensor:
    """Exact-erf GELU derivative Phi(a) + a phi(a)."""
    assert act.approximate == "none", "analytic gelu' below is for the exact GELU"
    return 0.5 * (1 + torch.erf(a / 2**0.5)) + a * torch.exp(-0.5 * a * a) / (2 * torch.pi) ** 0.5


def _leaky_relu_grad(act: nn.LeakyReLU, a: torch.Tensor) -> torch.Tensor:
    return torch.where(a > 0, torch.ones_like(a), torch.full_like(a, act.negative_slope))


_ACTIVATIONS = {
    "gelu": (nn.GELU, _gelu_grad),
    "leaky_relu": (nn.LeakyReLU, _leaky_relu_grad),
}


class RMSNorm(nn.Module):
    """RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps). No mean subtraction, no bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rms_jacobian(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Jacobian of RMSNorm without gamma at x, (B, d) -> (B, d, d).

    y_i = x_i / r with r = sqrt(mean(x^2) + eps), so d y_i / d x_j = (delta_ij - x_i x_j / (d r^2)) / r.
    """
    d = x.shape[-1]
    r = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()  # (B, 1)
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    return (eye - x[:, :, None] * x[:, None, :] / (d * r[:, :, None] ** 2)) / r[:, :, None]


class ResMLPBlock(nn.Module):
    """Pre-norm residual block: x + MLP(RMSNorm(x))."""

    def __init__(self, d_model: int, d_hidden: int | None = None, activation: str = "gelu"):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}")
        self.activation_name = activation
        act_cls, self._act_grad_fn = _ACTIVATIONS[activation]
        self.norm = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            act_cls(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))

    @torch.no_grad()
    def jacobian_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The two x-dependent factors of the Jacobian: J_rms (B, d, d), the inner RMSNorm Jacobian without
        gamma, and act'(a) (B, d_hidden), the activation slope at the pre-activation a = W1 RMS(x) + b1."""
        lin1, act, _ = self.mlp
        J_rms = _rms_jacobian(x, self.norm.eps)
        a = lin1(self.norm(x))  # pre-activation, (B, d_hidden)
        return J_rms, self._act_grad_fn(act, a)

    @torch.no_grad()
    def _mlp_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobian of the MLP branch alone (no residual skip): d MLP(RMS(x)) / dx, (B, d, d)."""
        J_rms, act_grad = self.jacobian_parts(x)
        lin1, _, lin2 = self.mlp
        return lin2.weight @ (
            act_grad[:, :, None] * lin1.weight @ (self.norm.weight[:, None] * J_rms)
        )  # (B, d, d)

    @torch.no_grad()
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic Jacobian df/dx of the block at x, (B, d) -> (B, d, d), J[b, i, j] = d f_i / d x_j."""
        return torch.eye(x.shape[-1], device=x.device, dtype=x.dtype) + self._mlp_jacobian(x)


class PreiNormResMLPBlock(ResMLPBlock):
    """Prei-norm residual block (norm before and after the MLP): x + RMSNorm2(MLP(RMSNorm(x)))."""

    def __init__(self, d_model: int, d_hidden: int | None = None, activation: str = "gelu"):
        super().__init__(d_model, d_hidden, activation)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        return x + self.norm2(self.mlp(self.norm(x)))

    @torch.no_grad()
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """J = I + diag(gamma2) J_rms2 dMLP/dx, with J_rms2 the RMSNorm2 Jacobian at the MLP output."""
        h = self.mlp(self.norm(x))  # (B, d) MLP output before norm2
        J_rms2 = _rms_jacobian(h, self.norm2.eps)
        J_inner = self.norm2.weight[:, None] * (J_rms2 @ self._mlp_jacobian(x))
        return torch.eye(x.shape[-1], device=x.device, dtype=x.dtype) + J_inner


_BLOCK_TYPES = {
    "pre": ResMLPBlock,
    "prei": PreiNormResMLPBlock,
}


class SyntheticModel(nn.Module):
    """Linear embed (intrinsic_dim -> d_model), n_blocks residual blocks, final RMSNorm, unembed (d_model -> n_classes).

    Blocks are assembled once here from _BLOCK_TYPES: norm_mode="pre" gives ResMLPBlock, "prei" gives PreiNormResMLPBlock.
    """

    def __init__(
        self,
        intrinsic_dim: int,
        d_model: int,
        n_classes: int,
        d_hidden: int | None = None,
        n_blocks: int = 2,
        activation: str = "gelu",
        norm_mode: str = "pre",
    ):
        super().__init__()
        if norm_mode not in _BLOCK_TYPES:
            raise ValueError(f"norm_mode must be one of {list(_BLOCK_TYPES)}, got {norm_mode!r}")
        blk_cls = _BLOCK_TYPES[norm_mode]
        self.embed = nn.Linear(intrinsic_dim, d_model)
        self.blocks = nn.Sequential(
            *[blk_cls(d_model, d_hidden, activation=activation) for _ in range(n_blocks)]
        )
        self.final_norm = RMSNorm(d_model)
        self.unembed = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.unembed(self.final_norm(self.blocks(self.embed(x))))  # logits
