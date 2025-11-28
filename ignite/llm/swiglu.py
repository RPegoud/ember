import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Module


class SwiGLU(Module):

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.model_dim = model_dim
        # recommended scaling to preserve the parameter count of a 2-layer FFN
        self.hidden_dim = int(2 * hidden_dim / 3)
        self.W = nn.Linear(self.model_dim, self.hidden_dim, bias=False)
        self.V = nn.Linear(self.model_dim, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, self.model_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.W(x))
        x_V = self.V(x)
        o = self.W2(swish * x_V)

        return o
