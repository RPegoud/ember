import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules import Module


class RMSNorm(Module):

    def __init__(
        self,
        feature_dims: tuple | int,
        eps: float = 1e-5,
        device: str = None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        if isinstance(feature_dims, int):
            feature_dims = (feature_dims,)
        self.feature_dims = tuple(feature_dims)

        self.weight = nn.Parameter(torch.empty(feature_dims, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def _norm(self, x: torch.Tensor, dims: tuple) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n_dims = len(self.feature_dims)
        dims = tuple(range(-n_dims, 0))  # last n_dims

        x_norm = self._norm(x.float(), dims).type_as(x)
        return x_norm * self.weight
