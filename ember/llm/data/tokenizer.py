from typing import Literal

import torch
from transformers import AutoTokenizer


class HFTokenizer:
    def __init__(self, model: str = "meta-llama/Llama-3.2-1B") -> None:
        self.base = AutoTokenizer.from_pretrained(model)
        self.eos_token_id = self.base.convert_tokens_to_ids(self.base.eos_token)
        self.base.add_special_tokens({"pad_token": "<|pad|>"})
        self.vocab_size = len(self.base)

    def __getattr__(self, attr):
        return getattr(self.base, attr)

    def __call__(
        self, x: torch.Tensor, mode: Literal["train", "inference"]
    ) -> torch.Tensor:
        original_side = self.base.padding_side

        try:
            if mode == "inference":
                self.base.padding_side = "left"
            else:
                self.base.padding_side = "right"

            return self.base(
                x,
                return_tensors="pt",
                return_attention_mask=False,
                padding="longest",
            )["input_ids"]

        finally:
            self.base.padding_side = original_side

    def encode(
        self, x: torch.Tensor, mode: Literal["train", "inference"]
    ) -> torch.Tensor:
        return self(x, mode)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            ndim = x.dim()
            x = x.tolist()
            if ndim > 1:
                return self.base.batch_decode(x)
            else:
                return self.base.decode(x)
        if isinstance(x, list):
            if any(isinstance(el, list) for el in x):  # nested list
                return self.base.batch_decode(x)
            else:
                return self.base.decode(x)
        if isinstance(x, int):
            return self.base.decode(x)
