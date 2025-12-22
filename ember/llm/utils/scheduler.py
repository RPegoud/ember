from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal["cosine", "constant"],
    max_steps: int,
    warmup_frac: float,
    use_warmup: bool,
) -> LambdaLR:
    """
    Unified scheduler factory.

    Args:
        optimizer (torch.optim.Optimizer): the PyTorch Optimizer
        scheduler_type (Literal["cosine", "constant"]): the type of scheduler
        max_steps (int): total number of optimizer steps
        warmup_frac (float): fraction of `max_steps` to use for warmup (0.0 to 1.0)
        use_warmup (bool): master toggle for warmup

    Returns:
        LambdaLR: the PyTorch Scheduler
    """

    if use_warmup:
        num_warmup_steps = int(max_steps * warmup_frac)
    else:
        num_warmup_steps = 0

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_steps,
        )

    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
