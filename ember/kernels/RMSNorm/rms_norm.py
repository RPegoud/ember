import math

import torch
import triton
import triton.language as tl

from ..kernel_utils import ensure_contiguous


@triton.jit
def rms_norm_fwd_kernel(
    X_ptr,
    X_row_stride,
    Y_ptr,
    Y_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_id * X_row_stride
    Y_ptr += row_id * Y_row_stride
    RSTD_ptr += row_id * RSTD_row_stride

    x = tl.load(
        pointer=X_ptr + col_offsets,
        mask=mask,
        other=0.0,
    )
    w = tl.load(pointer=W_ptr + col_offsets, mask=mask, other=0.0)

    # cast to float32 for computation then cast back to original type
    x_dtype = x.dtype
    x_f32 = x.to(tl.float32)

    mean_square = tl.sum(x_f32 * x_f32, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)

    # cache rms for backward (small compared to X and saves *, sum, /, sqrt)
    tl.store(pointer=RSTD_ptr, value=rstd)

    x_f32 = x_f32 * rstd
    x = x_f32.to(x_dtype)

    y = x * w
    y = y.to(x_dtype)

    tl.store(
        pointer=Y_ptr + col_offsets,
        value=y,
        mask=mask,
    )


@triton.jit
def rmsnorm_bwd_kernel(
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    dX_ptr,
    dX_row_stride,
    dY_ptr,
    dY_row_stride,
    W_ptr,
    dW_ptr,
    dW_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_start * X_row_stride
    dX_ptr += row_start * dX_row_stride
    dY_ptr += row_start * dY_row_stride
    RSTD_ptr += row_start

    dW_ptr += row_block_id * dW_row_stride

    # W is shared across the batch, load full row once
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # accumulate gradients across the batch for multiple rows per program
    for _ in range(row_start, row_end):
        dY_row = tl.load(pointer=dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(pointer=X_ptr + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(pointer=RSTD_ptr)  # get cached RSTD

        X_row = X_row.to(tl.float32)
        m = (dY_row * W_row).to(tl.float32)

        # dX_row = rstd_row * W_row
        dX_row = rstd_row * (
            m - (1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row) * X_row
        )
        dW_row += dY_row * (X_row * rstd_row)

        tl.store(pointer=dX_ptr + col_offsets, value=dX_row.to(X_dtype), mask=mask)

        # move pointers to the next row / element
        X_ptr += X_row_stride
        dX_ptr += dX_row_stride
        dY_ptr += dY_row_stride
        RSTD_ptr += RSTD_row_stride

    tl.store(pointer=dW_ptr + col_offsets, value=dW_row, mask=mask)


class EmberRMSNorm(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        n_rows, n_cols = x.shape
        y = torch.empty_like(x, dtype=x.dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        rms_norm_fwd_kernel[(n_rows,)](
            X_ptr=x,
            X_row_stride=x.stride(0),
            Y_ptr=y,
            Y_row_stride=y.stride(0),
            W_ptr=w,
            RSTD_ptr=rstd,
            RSTD_row_stride=rstd.stride(0),
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.save_for_backward(x, w, rstd)

        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY: torch.Tensor):
        X, W, rstd = ctx.saved_tensors
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        n_rows, n_cols = dY.shape

        sm_count = 1
        if X.device.type == "cuda":
            sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count

        partial_dW = torch.empty(
            (sm_count, n_cols), dtype=torch.float32, device=W.device
        )

        rows_per_program = math.ceil(n_rows / sm_count)
        dX = torch.empty_like(dY)

        rmsnorm_bwd_kernel[(sm_count,)](
            X_ptr=X,
            X_row_stride=X.stride(0),
            X_dtype=X.dtype,
            dX_ptr=dX,
            dX_row_stride=dX.stride(0),
            dY_ptr=dY,
            dY_row_stride=dY.stride(0),
            W_ptr=W,
            dW_ptr=partial_dW,
            dW_row_stride=partial_dW.stride(0),
            RSTD_ptr=rstd,
            RSTD_row_stride=rstd.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            rows_per_program=rows_per_program,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        dX = dX.view(*shape)
        dW = partial_dW.sum(dim=0).to(W.dtype)

        return dX, dW
