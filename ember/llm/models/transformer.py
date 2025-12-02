from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim

from ...core.weight_init import init_weights
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
        learning_rate: Optional[float] = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.n_attn_blocks = n_attn_blocks
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

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids = batch
        inputs = input_ids[:, :-1].contiguous()  # shift left
        targets = input_ids[:, 1:].contiguous()  # shift right

        logits = self(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("Train loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for attn_block in self.attn_blocks:
            h = attn_block(h)
        h = self.norm(h)
        # weight tying (use embedding weights as output weights)
        logits = F.linear(h, weight=self.embed.weight)
        return logits
