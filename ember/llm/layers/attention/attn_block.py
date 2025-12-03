from typing import Optional

import torch
import torch.nn as nn

from ...data.kv_cache import LayerKVCache
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
        self.mlp = SwiGLU(model_dim, hidden_dim)
        self.norm = RMSNorm(feature_dims=model_dim)
        self.attn = attn_module(**attn_kwargs)

    def forward(
        self, x: torch.Tensor, layer_cache: Optional[LayerKVCache] = None
    ) -> torch.Tensor:
        norm = self.norm(x)
        attn = self.attn(norm, layer_cache) + x
        norm_attn = self.norm(attn)
        return self.mlp(norm_attn) + attn
