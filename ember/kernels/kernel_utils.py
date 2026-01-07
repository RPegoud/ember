import functools
from itertools import product

import torch
import triton


def get_autotune_config(
    block_sizes: list[int], num_warps: list[int]
) -> list[triton.Config]:
    return [
        triton.Config(kwargs={"BLOCK_SIZE": bs}, num_warps=nw)
        for (bs, nw) in list(product(block_sizes, num_warps))
    ]


def ensure_contiguous(fn):
    # source: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/utils.py#L32
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper
