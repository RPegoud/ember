from typing import Literal, Protocol

import torch


class Tokenizer(Protocol):
    vocab_size: int

    def __call__(
        self, x: torch.Tensor, mode: Literal["train", "inference"]
    ) -> torch.Tensor: ...

    def encode(
        self, x: torch.Tensor, mode: Literal["train", "inference"]
    ) -> torch.Tensor: ...

    def decode(self, x: torch.Tensor) -> torch.Tensor: ...
