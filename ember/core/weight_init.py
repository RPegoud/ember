import torch
import torch.nn as nn
from torch.nn import init

from ember.llm.rmsnorm import RMSNorm


@torch.no_grad()
def init_weights(module: nn.Module) -> None:
    """
    Global initialisation strategy:
        - nn.Linear: random normal, zero-mean, std=0.02
        - nn.Embedding: random normal, zero-mean, std=0.02
        - nn.LayerNorm, nn.RMSNorm, custom RMSNorm: ones, zero bias
    """

    if isinstance(module, nn.Linear):
        init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, RMSNorm)):
        init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            init.zeros_(module.bias)
