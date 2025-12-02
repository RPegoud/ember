import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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

        self.fused_qkv = nn.Linear(model_dim, model_dim + self.kv_head_dim * 2)
        self.out_proj = nn.Linear(model_dim, model_dim)

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

        # duplicate shared heads to align q, k, v shapes
        k, v = map(self.duplicate_heads, (k, v))

        attn_outputs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_outputs = rearrange(attn_outputs, "B NH S HD -> B S (NH HD)")

        return self.out_proj(attn_outputs)
