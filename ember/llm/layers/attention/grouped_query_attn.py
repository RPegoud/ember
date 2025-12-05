from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...types import LayerCache
from ..embeddings import RoPE, apply_rotary_pos_emb


class GroupedQueryAttn(nn.Module):

    def __init__(
        self,
        model_dim: int,
        n_query_heads: int,  # number of query heads
        n_query_groups: int,  # number of groups of query heads sharing single k/v heads
        rope_theta: int = 50_000,
    ):
        """
        Grouped Query Attention, similarly to Llama3.

        Args:
            model_dim (int): the size of the feature dimension
            n_heads (int): the number of query attention heads
            n_query_groups (int): the number of query heads sharing single k/v heads
            rope_theta (int): the theta parameter of RoPE embeddings

        Einsum notation:
            - `B`: batch size
            - `S`: sequence length
            - `NH`: number of heads
            - `HD`: head dimension
        """
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

        self.kv_dim = model_dim // self.kv_heads_per_q_head
        self.kv_head_dim = self.kv_dim // self.n_query_groups

        self.fused_qkv = nn.Linear(model_dim, model_dim + self.kv_dim * 2)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.rope = RoPE(dim=self.kv_head_dim, base=rope_theta)

    @property
    def cache_requirements(self) -> Mapping[str, int]:
        """
        Exposes the number of KV heads and their dimensions for the KV cache.

        Returns:
            Mapping[str, int]:
                - `n_heads` (int): the number of KV heads
                - `head_dim` (int): the dimension of a single head
        """
        return {
            "n_heads": self.n_query_groups,
            "head_dim": self.kv_head_dim,
        }

    def duplicate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates duplicate views of shared k/v heads to align
        with queries shape. The created views hold no space in memory.
        """
        x = x.unsqueeze(2).expand(-1, -1, self.kv_heads_per_q_head, -1, -1)
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        return x

    def forward(
        self, x: torch.Tensor, layer_cache: Optional[LayerCache] = None
    ) -> torch.Tensor:
        # --- QKV Projections ---
        fused_proj = self.fused_qkv(x)
        q, k, v = torch.split(
            fused_proj, [self.model_dim, self.kv_dim, self.kv_dim], dim=-1
        )

        # --- Reshape to Multi-Head ---
        q = rearrange(q, "B S (NH HD) -> B NH S HD", NH=self.n_query_heads)
        k, v = map(
            lambda x: rearrange(x, "B S (NH HD) -> B NH S HD", NH=self.n_query_groups),
            (k, v),
        )

        # --- Positional Embeddings ---
        cos, sin = self.rope(
            q, offset=layer_cache.parent_cache.current_len if layer_cache else 0
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # --- KV Cache Update ---
        if layer_cache is not None:
            k, v = layer_cache.update(k, v)

        # --- Duplicate heads to align QKV shapes ---
        k, v = map(self.duplicate_heads, (k, v))

        # --- Attention ---
        attn_outputs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_outputs = rearrange(attn_outputs, "B NH S HD -> B S (NH HD)")

        return self.out_proj(attn_outputs)
