from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...types import LayerCache
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
        rope_theta: int = 50_000,
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

        self.q_up_proj = nn.Linear(latent_dim, model_dim)
        self.k_up_proj = nn.Linear(latent_dim, model_dim)
        self.v_up_proj = nn.Linear(latent_dim, model_dim)
        # self.fused_qkv_up_proj = nn.Linear(latent_dim * 3, model_dim * 3)

        # Positional projections (decoupled RoPE)
        self.q_pos_proj = nn.Linear(latent_dim, pos_dim)
        self.k_pos_proj = nn.Linear(
            model_dim, self.pos_head_dim
        )  # project to a single pos head

        # Output projections
        self.o_proj = nn.Linear(model_dim, model_dim)

        self.rope = RoPE(dim=self.pos_head_dim, base=rope_theta)

    @property
    def cache_requirements(self) -> Mapping[str, int]:
        return {
            "n_heads": 1,
            "head_dim": self.latent_dim + self.pos_head_dim,
        }

    def forward(
        self, x: torch.Tensor, layer_cache: Optional[LayerCache] = None
    ) -> torch.Tensor:
        B, S, _ = x.shape

        # --- Latent projection ---
        latent_qkv = self.fused_qkv_down_proj(x)
        latent_q, latent_k, latent_v = torch.split(
            latent_qkv, int(self.latent_dim), dim=-1
        )

        # --- Positional projection (new token) ---
        pos_k = self.k_pos_proj(x)
        pos_k_rope = pos_k.unsqueeze(1)  # dummy head dim => [B, 1, S, D]
        offset = layer_cache.parent_cache.current_len if layer_cache else 0
        cos, sin = self.rope(pos_k_rope, offset=offset)

        # rotate only the keys
        _, pos_k = apply_rotary_pos_emb(pos_k_rope, pos_k_rope, cos, sin)

        # --- KV Caching ---
        if layer_cache is not None:
            # prepare kv payloads, reshape to [B, 1, S, D] to meet cache requirements
            k_payload = torch.cat([latent_k.unsqueeze(1), pos_k], dim=-1)
            padding = torch.zeros(
                (B, 1, S, self.pos_head_dim), device=x.device, dtype=x.dtype
            )  # pad v to match k's dimension
            v_payload = torch.cat([latent_v.unsqueeze(1), padding], dim=-1)

            # update and retireve history
            cached_k, cached_v = layer_cache.update(k_payload, v_payload)

            # squeeze dummy head dim, unpack to latent and pos tensors
            cached_k, cached_v = map(lambda x: x.squeeze(1), (cached_k, cached_v))
            latent_k_hist, latent_v_hist = map(
                lambda x: x[..., : self.latent_dim], (cached_k, cached_v)
            )
            pos_k_hist = cached_k[..., self.latent_dim :]

        else:  # training mode (no cache)
            latent_k_hist = latent_k
            latent_v_hist = latent_v
            pos_k_hist = pos_k.squeeze(1)

        # --- Up projection ---
        q = self.q_up_proj(latent_q)
        k = self.k_up_proj(latent_k_hist)
        v = self.v_up_proj(latent_v_hist)

        # reshape to multi-head
        q, k, v = map(
            lambda x: rearrange(x, "B S (NH HD) -> B NH S HD", NH=self.n_heads),
            (q, k, v),
        )

        # --- Combine with Positional embeddings ---
        # Q Positional projection (current step)
        pos_q = self.q_pos_proj(latent_q)
        pos_q = rearrange(pos_q, "B S (NH HD) -> B NH S HD", NH=self.n_heads)
        cos, sin = self.rope(pos_q, offset=offset)

        pos_q, _ = apply_rotary_pos_emb(pos_q, pos_q, cos, sin)
        # K Positional projection (full history), broadcast to multi-head
        pos_k_hist = pos_k_hist.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # --- Attention ---
        # Merge content and positional heads
        q = torch.cat([q, pos_q], dim=-1)
        k = torch.cat([k, pos_k_hist], dim=-1)

        # Attention and output projection
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = rearrange(attn, "B NH S HD -> B S (NH HD)", NH=self.n_heads)

        return self.o_proj(attn)
