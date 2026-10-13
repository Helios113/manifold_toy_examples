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
