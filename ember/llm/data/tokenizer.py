import torch
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model: str = "meta-llama/Llama-3.2-1B") -> None:
        self.base = AutoTokenizer.from_pretrained(model)
        self.eos_token_id = self.base.convert_tokens_to_ids(self.base.eos_token)
        self.base.pad_token = self.base.eos_token
        self.vocab_size = len(self.base)

    def __getattr__(self, attr):
        return getattr(self.base, attr)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(
            x,
            return_tensors="pt",
            return_attention_mask=False,
            padding="longest",
        )["input_ids"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

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
