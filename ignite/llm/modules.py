import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from torch.nn.modules import Module


class RMSNorm(Module):
    def __init__(
        self,
        feature_dims: tuple | int,
        eps: float = 1e-5,
        device: str = None,
        dtype=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        if isinstance(feature_dims, int):
            feature_dims = (feature_dims,)
        self.feature_dims = tuple(feature_dims)

        self.weight = nn.Parameter(torch.empty(feature_dims, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n_dims = len(self.feature_dims)
        dims = tuple(range(-n_dims, 0))  # last n_dims

        var = x.pow(2).mean(dim=dims, keepdim=True) + self.eps
        x_norm = x * torch.rsqrt(var)
        rmsnorm = x_norm * self.weight

        return rmsnorm


class RoPE(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(x: torch.Tensor) -> torch.Tensor: ...


class SwiGLU(Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        # recommended scaling to preserve the parameter count of a 2-layer FFN
        self.hidden_dim = int(2 * hidden_dim / 3)
        self.W = nn.Linear(self.model_dim, self.hidden_dim, bias=False)
        self.V = nn.Linear(self.model_dim, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, self.model_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        map(init.xavier_normal, (self.W.weight, self.W2.weight, self.V.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.W(x))
        x_V = self.V(x)
        o = self.W2(swish * x_V)

        return o


class GroupedQueryAttn(nn.Module):
    # TODO: add masking
    def __init__(
        self,
        model_dim: int,
        n_query_heads: int,
        n_query_groups: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.fused_qkv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_proj = self.fused_qkv(x)

        q, k, v = torch.split(
            fused_proj, [self.model_dim, self.kv_head_dim, self.kv_head_dim], dim=-1
        )

        # split q, k, v in different heads
        q = rearrange(q, "B S (NH HD) -> B NH S HD", NH=self.n_query_heads)
        k = rearrange(k, "B S (NH HD) -> B NH HD S", NH=self.n_query_groups)
        v = rearrange(v, "B S (NH HD) -> B NH S HD", NH=self.n_query_groups)

        # create dupplicate views of shared k/v heads to align shapes (no additional memory)
        k = repeat(
            k, "B NH HD S -> B (NH repeat) HD S", repeat=self.kv_heads_per_q_head
        )
        v = repeat(
            v, "B NH S HD -> B (NH repeat) S HD", repeat=self.kv_heads_per_q_head
        )

        sim = q @ k / math.sqrt(self.q_head_dim)
        attn_scores = torch.softmax(sim, dim=-1)
        attn_outputs = rearrange(attn_scores @ v, "B NH S HD -> B S (NH HD)")
        return self.out_proj(attn_outputs)
