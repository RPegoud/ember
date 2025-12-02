from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..embeddings import RoPE, apply_rotary_pos_emb


class MultiHeadLatentAttn(nn.Module):
    """
    Multi-Head Latent Attention.

    Einsum notation:
        - `B`: batch size
        - `S`: sequence length
        - `NH`: number of heads
        - `HD`: head dimension
    """

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        pos_dim: int,
        n_heads: int,
        rope_theta: Optional[int] = 50_000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.pos_dim = pos_dim
        self.n_heads = n_heads

        self.head_dim = model_dim // n_heads
        self.pos_head_dim = pos_dim // n_heads

        # Content projections
        self.fused_qkv_down_proj = nn.Linear(model_dim, latent_dim * 3)
        self.fused_qkv_up_proj = nn.Linear(latent_dim * 3, model_dim * 3)

        # Positional projections (decoupled RoPE)
        self.q_pos_proj = nn.Linear(latent_dim, pos_dim)
        self.k_pos_proj = nn.Linear(
            model_dim, self.pos_head_dim
        )  # project to a single pos head

        # Output projections
        self.o_proj = nn.Linear(model_dim, model_dim)

        self.rope = RoPE(dim=self.pos_head_dim, base=rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Latent projections
        latent_qkv = self.fused_qkv_down_proj(x)
        latent_q, *_ = torch.split(latent_qkv, int(self.latent_dim), dim=-1)

        # Up projections, reshape to multi-head
        fused_qkv = self.fused_qkv_up_proj(latent_qkv)
        q, k, v = torch.split(fused_qkv, int(self.model_dim), dim=-1)  # cache kv
        q, k, v = map(
            lambda x: rearrange(x, "B S (NH HD) -> B NH S HD", NH=self.n_heads),
            (q, k, v),
        )

        # Positional Embeddings
        pos_q = self.q_pos_proj(latent_q)
        pos_k = self.k_pos_proj(x)

        pos_q = rearrange(pos_q, "B S (NH HD) -> B NH S HD", NH=self.n_heads)
        pos_k = rearrange(pos_k, "B S (NH HD) -> B NH S HD", NH=1)

        cos, sin = self.rope(pos_q)
        cos, sin = map(lambda x: x.transpose(0, 2), (cos, sin))
        pos_q, pos_k = apply_rotary_pos_emb(pos_q, pos_k, cos, sin)
        pos_k = pos_k.expand(-1, self.n_heads, -1, -1)  # broadcast to all heads

        # Merge content and positional heads
        q = torch.cat([q, pos_q], dim=-1)
        k = torch.cat([k, pos_k], dim=-1)

        # Attention and output projection
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = rearrange(attn, "B NH S HD -> B S (NH HD)", NH=self.n_heads)

        return self.o_proj(attn)
