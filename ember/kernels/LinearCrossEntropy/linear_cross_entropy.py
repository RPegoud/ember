import math

import torch
import triton
import triton.language as tl

from ..kernel_utils import ensure_contiguous


@triton.jit
def linear_cross_entropy_fwd_kernel(
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    W_col_stride,
    Y_ptr,
    dX_ptr,
    dX_row_stride,
    dW_ptr,
    dW_row_stride,
    dW_col_stride,
    Loss_ptr,
    D: tl.constexpr,
    V: tl.constexpr,
    D_BLOCK: tl.constexpr,
    V_BLOCK: tl.constexpr,
    ignore_index: tl.constexpr,
):
    # --- Setup ---
    row_id = tl.program_id(axis=0).to(tl.int64)
    d_tile_offs = tl.arange(0, D_BLOCK)
    v_tile_offs = tl.arange(0, V_BLOCK)

    m = -float("inf")  # running max
    d = 0.0  # running exp sum

    # --- Pointer Logic ---
    X_ptr += row_id * X_row_stride
    dX_ptr += row_id * dX_row_stride
    Y_ptr += row_id
    Loss_ptr += row_id

    # --- Pre-compute the Y logit ---
    Y = tl.load(pointer=Y_ptr)

    # if Y should be ignored, zero the loss and return early
    if Y == ignore_index:
        tl.store(Loss_ptr, 0.0)
        return

    Y_logit_acc = 0.0

    # load the X and W tile, accumulate the Y logit
    for d_idx in tl.range(0, D, D_BLOCK):
        d_tile_mask = (d_idx + d_tile_offs) < D
        X_tile = tl.load(X_ptr + d_idx + d_tile_offs, mask=d_tile_mask, other=0.0)

        W_target_tile_ptr = (
            W_ptr + (d_idx + d_tile_offs) * W_row_stride + Y * W_col_stride
        )
        W_target_tile = tl.load(W_target_tile_ptr, mask=d_tile_mask, other=0.0)

        Y_logit_acc += tl.sum(X_tile.to(tl.float32) * W_target_tile.to(tl.float32))

    # --- Forward Pass ---
    # iterate through X by tiles and W by blocks
    # compute X@W, accumulate the running sum and compute the max
    for v_idx in tl.range(0, V, V_BLOCK):
        W_block_ptr = tl.make_block_ptr(
            base=W_ptr,
            shape=(D, V),
            strides=(W_row_stride, W_col_stride),
            offsets=(0, v_idx),
            block_shape=(D_BLOCK, V_BLOCK),
            order=(0, 1),
        )

        # compute the logits: l_tile = X_tile @ W_tile
        acc = tl.zeros((1, V_BLOCK), dtype=tl.float32)
        for d_idx in tl.range(0, D, D_BLOCK):
            tile_offs = d_idx + d_tile_offs
            tile_mask = tile_offs < D

            X_tile = tl.load(X_ptr + tile_offs, mask=tile_mask, other=0.0)
            W_block = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option="zero")
            acc += tl.dot(X_tile[None, :], W_block)

            W_block_ptr = W_block_ptr.advance((D_BLOCK, 0))

        # update running sum and max
        m_tile = tl.max(acc)
        new_m = tl.maximum(m, m_tile)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(acc - new_m))
        m = new_m

    lse = tl.log(d)
    loss = m - Y_logit_acc + lse

    tl.store(pointer=Loss_ptr, value=loss)

    # --- Backward Pass ---

    # 1: Compute the normalised probabilities P
    for v_idx in tl.range(0, V, V_BLOCK):
        W_block_ptr = tl.make_block_ptr(
            base=W_ptr,
            shape=(D, V),
            strides=(W_row_stride, W_col_stride),
            offsets=(0, v_idx),
            block_shape=(D_BLOCK, V_BLOCK),
            order=(0, 1),
        )

        P_tile = tl.zeros((1, V_BLOCK), dtype=tl.float32)
        for d_idx in tl.range(0, D, D_BLOCK):
            tile_offs = d_idx + d_tile_offs
            tile_mask = tile_offs < D

            X_tile = tl.load(X_ptr + tile_offs, mask=tile_mask, other=0.0)
            W_block = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option="zero")
            P_tile += tl.dot(X_tile[None, :].to(tl.float32), W_block.to(tl.float32))

            W_block_ptr = W_block_ptr.advance((D_BLOCK, 0))

        P_tile = tl.exp(P_tile - lse)  # normalise
        P_tile -= tl.where(
            v_idx + v_tile_offs == Y, 1, 0
        )  # subtract 1 when logit = label

        # 2: Compute the gradients:
        # dX = (P - Y) . W^T
        # dW = X^T . (P - Y)
        W_T_block_ptr = tl.make_block_ptr(  # logically transpose W
            base=W_ptr,
            shape=(V, D),
            strides=(W_col_stride, W_row_stride),
            offsets=(v_idx, 0),
            block_shape=(V_BLOCK, D_BLOCK),
            order=(1, 0),  # W is row major => W^T is column major
        )

        for d_idx in tl.range(0, D, D_BLOCK):
            # 2.1: Accumulate dX
            tile_offs = d_idx + d_tile_offs
            tile_mask = tile_offs < D

            W_T_block = tl.load(
                W_T_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            dX_partial = tl.dot(P_tile, W_T_block).reshape(D_BLOCK)
            tl.atomic_add(
                pointer=dX_ptr + d_idx + d_tile_offs,
                val=dX_partial,
                sem="relaxed",  # the order of adds across threads does not matter
            )

            W_T_block_ptr = W_T_block_ptr.advance((0, D_BLOCK))

            # 2.2: Accumulate dW
            dW_tile_ptr = dW_ptr + (
                (d_idx + d_tile_offs)[:, None] * dW_row_stride
                + (v_idx + v_tile_offs)[None, :] * dW_col_stride
            )

            dW_mask = ((d_idx + d_tile_offs)[:, None] < D) & (
                (v_idx + v_tile_offs)[None, :] < V
            )

            X_T_tile = tl.load(X_ptr + tile_offs, mask=tile_mask, other=0.0)[:, None]

            dW_partial = X_T_tile.to(tl.float32) * P_tile
            tl.atomic_add(
                pointer=dW_tile_ptr, val=dW_partial, mask=dW_mask, sem="relaxed"
            )
