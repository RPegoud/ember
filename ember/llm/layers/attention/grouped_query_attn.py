from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..embeddings import RoPE, apply_rotary_pos_emb


class GroupedQueryAttn(nn.Module):
    """
    Grouped Query Attention.

    Einsum notation:
        - `B`: batch size
        - `S`: sequence length
        - `NH`: number of heads
        - `HD`: head dimension
    """

    def __init__(
        self,
        model_dim: int,
        n_query_heads: int,
        n_query_groups: int,
        rope_theta: Optional[int] = 50_000,
    ):
        super().__init__()
        assert (
            model_dim % n_query_heads == 0
        ), f"{model_dim=} is not divisible by {n_query_heads=}"
        assert (
            n_query_heads % n_query_groups == 0
        ), f"{n_query_heads=} is not divisible by {n_query_groups=}"

        self.model_dim = model_dim
        self.n_query_heads = n_query_heads
        self.n_query_groups = n_query_groups
        self.kv_heads_per_q_head = n_query_heads // n_query_groups

        self.q_head_dim = model_dim // n_query_heads
        self.kv_head_dim = model_dim // self.kv_heads_per_q_head
        self.kv_single_head_dim = self.kv_head_dim // self.n_query_groups

        self.fused_qkv = nn.Linear(model_dim, model_dim + self.kv_head_dim * 2)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.rope = RoPE(dim=self.kv_single_head_dim, base=rope_theta)

    def duplicate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates duplicate views of shared k/v heads to align
        with queries shape. The created views hold no space in memory.
        """
        x = x.unsqueeze(2).expand(-1, -1, self.kv_heads_per_q_head, -1, -1)
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_proj = self.fused_qkv(x)

        q, k, v = torch.split(
            fused_proj, [self.model_dim, self.kv_head_dim, self.kv_head_dim], dim=-1
        )

        # split q, k, v in different heads
        q = rearrange(q, "B S (NH HD) -> B NH S HD", NH=self.n_query_heads)
        k, v = map(
            lambda x: rearrange(x, "B S (NH HD) -> B NH S HD", NH=self.n_query_groups),
            (k, v),
        )

        cos, sin = self.rope(q)
        cos, sin = map(lambda x: x.transpose(0, 2), (cos, sin))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # duplicate shared heads to align q, k, v shapes
        k, v = map(self.duplicate_heads, (k, v))

        attn_outputs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_outputs = rearrange(attn_outputs, "B NH S HD -> B S (NH HD)")

        return self.out_proj(attn_outputs)
