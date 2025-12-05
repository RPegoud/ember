from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Cache(Protocol):
    k_cache: list[torch.tensor]
    v_cache: list[torch.tensor]
    current_len: int
    n_layers: int

    def store(
        self, k: torch.Tensor, v: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def step(self) -> None: ...

    def initialize_prefill(self, seq_len: int) -> None: ...


@runtime_checkable
class LayerCache(Protocol):
    parent_cache: Cache
    layer_idx: int

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
