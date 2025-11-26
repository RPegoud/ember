import pytest
import torch

from ignite.llm import RMSNorm


@pytest.mark.parametrize(
    "shape,f_dims",
    [
        ((1,), 1),
        ((3, 4), 4),
        ((2, 5, 16), 16),
        ((3, 4, 8, 8), (8, 8)),
        ((2, 3, 7, 11, 13), (11, 13)),
    ],
)
def test_rmsnorm_shapes(shape, f_dims):
    x = torch.randn(shape)
    norm = RMSNorm(f_dims)
    y = norm(x)
    assert y.shape == x.shape
