from typing import Protocol

import torch
import torch.nn as nn


class Sampler(nn.Module, Protocol):
    temperature: float

    def __call__(self, temperature: float, *args) -> torch.Tensor: ...
