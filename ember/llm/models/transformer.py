from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from ...core.weight_init import init_weights
from ..data import KVCache, LayerKVCache
from ..layers import AttentionBlock, RMSNorm
from ..types import Sampler, Tokenizer


class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        hidden_dim: int,
        max_seq_len: int,
        n_attn_blocks: int,
        attention: DictConfig,
        pad_token_id: int,
        device: str,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.n_attn_blocks = n_attn_blocks
        self.pad_token_id = pad_token_id
        self.device = device

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    model_dim,
                    hidden_dim,
                    instantiate(attention, model_dim=model_dim),
                )
                for _ in range(n_attn_blocks)
            ]
        )
        self.norm = RMSNorm(feature_dims=model_dim)

        self.apply(init_weights)

    @property
    def cache_config(self) -> Mapping[str, int]:
        return self.attn_blocks[0].attn.cache_requirements

    def get_causal_loss(self, batch: torch.Tensor) -> float:
        input_ids = batch
        inputs = input_ids[:, :-1].contiguous()  # shift left

        targets = input_ids.clone()[:, 1:].contiguous()  # shift right
        targets[targets == self.pad_token_id] = -100  # mask loss for <|pad|> tokens

        logits = self.forward(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

    @torch.compile
    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        h = self.embed(x)
        for layer_idx, attn_block in enumerate(self.attn_blocks):
            layer_cache = LayerKVCache(cache, layer_idx) if cache else None
            h = attn_block(h, layer_cache)
        h = self.norm(h)
        logits = F.linear(
            h, weight=self.embed.weight
        )  # weight tying (use embedding weights as output weights)
        return logits

    @torch.inference_mode
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        sampler: Sampler,
        tokenizer: Tokenizer,
    ) -> torch.Tensor:
        cache_config: dict = self.cache_config

        indices = tokenizer.encode(prompts, mode="inference").to(self.device)
        B, S = indices.shape
        assert (
            S + max_new_tokens <= self.max_seq_len
        ), f"Got input with sequence length {S}, generating {max_new_tokens=} will exceed {self.max_seq_len=}"

        cache = KVCache(
            n_layers=self.n_attn_blocks,
            max_batch_size=B,
            max_seq_len=S + max_new_tokens,
            n_heads=cache_config["n_heads"],
            head_dim=cache_config["head_dim"],
            device=self.device,
            dtype=indices.dtype,  # TODO: is this the right way to pass dtype?
        )
        finished = torch.zeros((B,), dtype=torch.bool, device=indices.device)

        logits = self.forward(indices, cache)  # prefill cache
        cache.initialize_prefill(S)

        max_tokens = S + max_new_tokens
        while cache.current_len <= max_tokens:
            next_tokens = sampler(logits[:, -1, :])
            indices = torch.cat([indices, next_tokens], dim=-1)

            is_eos = next_tokens.squeeze(-1) == tokenizer.eos_token_id
            finished = finished | is_eos
            if finished.all():
                break

            logits = self.forward(next_tokens, cache=cache)
            cache.step()

        # remove tokens past <eos>
        output_strings = []
        for i in range(B):
            seq = indices[i].tolist()
            try:
                eos_idx = seq.index(tokenizer.eos_token_id)
                seq = seq[:eos_idx]
            except ValueError:
                pass
            output_strings.append(tokenizer.decode(seq))

        return output_strings
