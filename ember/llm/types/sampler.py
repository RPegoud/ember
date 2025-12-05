from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Sampler(Protocol):
    temperature: float

    def __call__(self, temperature: float, *args) -> torch.Tensor: ...
