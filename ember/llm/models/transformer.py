from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim

from ...core.weight_init import init_weights
from ..data.kv_cache import KVCache, LayerKVCache
from ..layers import AttentionBlock, RMSNorm


class Transformer(L.LightningModule):

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        hidden_dim: int,
        attn_module: nn.Module,
        attn_kwargs: dict[str, int],
        n_attn_blocks: int,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.n_attn_blocks = n_attn_blocks
        self.attn_kwargs = attn_kwargs
        self.lr = learning_rate
        attn_kwargs["model_dim"] = model_dim

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(model_dim, hidden_dim, attn_module, attn_kwargs)
                for _ in range(n_attn_blocks)
            ]
        )
        self.norm = RMSNorm(feature_dims=model_dim)

        self.apply(init_weights)

        print(self.parameters)
        print(
            """Parameter count: """
            f"""{sum(p.numel() for p in self.parameters() if p.requires_grad):.2e}"""
        )

    @property
    def cache_config(self):
        return self.attn_blocks[0].attn.cache_requirements

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        # TODO: ensure right padding
        input_ids = batch
        inputs = input_ids[:, :-1].contiguous()  # shift left

        targets = input_ids.clone()[:, 1:].contiguous()  # shift right
        targets[targets == self.tokenizer.pad_id] = -100  # mask loss for pad tokens

        logits = self(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("Train loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> None:
        return optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        h = self.embed(x)
        for layer_idx, attn_block in enumerate(self.attn_blocks):
            layer_cache = LayerKVCache(cache, layer_idx) if cache else None
            h = attn_block(h, layer_cache)
        h = self.norm(h)
        # weight tying (use embedding weights as output weights)
        logits = F.linear(h, weight=self.embed.weight)
        return logits

    @torch.inference_mode
    def generate(
        self,
        indices: torch.Tensor,
        max_new_tokens: int,
        sampler: nn.Module,
        tokenizer: callable,
    ) -> torch.Tensor:
        # TODO: ensure left padding
        cache_config: dict = self.cache_config
        B, S = indices.shape

        cache = KVCache(
            n_layers=self.n_attn_blocks,
            max_batch_size=B,
            max_seq_len=S + max_new_tokens,
            n_heads=cache_config["n_heads"],
            head_dim=cache_config["head_dim"],
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
