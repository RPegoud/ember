from typing import Mapping, Optional, Protocol, runtime_checkable

import torch

from .cache import LayerCache


@runtime_checkable
class Attention(Protocol):
    @property
    def cache_requirements(self) -> Mapping[str, int]: ...

    def __call__(
        self, x: torch.Tensor, layer_cache: Optional[LayerCache]
    ) -> torch.Tensor: ...
