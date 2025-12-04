from typing import Mapping, Optional, Protocol

import torch
import torch.nn as nn

from .cache import LayerCache


class Attention(nn.Module, Protocol):
    @property
    def cache_requirements(self) -> Mapping[str, int]: ...

    def __call__(
        self, x: torch.Tensor, layer_cache: Optional[LayerCache]
    ) -> torch.Tensor: ...
