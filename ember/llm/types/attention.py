from typing import Mapping, Optional, Protocol

import torch

from .cache import LayerCache


class Attention(Protocol):
    @property
    def cache_requirements(self) -> Mapping[str, int]: ...

    def __call__(
        self, x: torch.Tensor, layer_cache: Optional[LayerCache]
    ) -> torch.Tensor: ...
