import torch
import torch.nn as nn

from ..mlp import SwiGLU
from ..norm import RMSNorm


class AttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        attn_module: nn.Module,
        attn_kwargs: dict[str, int],
    ):
        super().__init__()
        self.model_dim = model_dim
        self.embeddings = ...
        self.mlp = SwiGLU(model_dim, hidden_dim)
        self.norm = RMSNorm(feature_dims=model_dim)
        self.attn = attn_module(**attn_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.norm(x)
        attn = self.attn(norm) + x
        norm_attn = self.norm(attn)
        return self.mlp(norm_attn) + attn
