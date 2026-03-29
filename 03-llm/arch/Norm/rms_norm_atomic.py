"""
RMS Norm with atomic_add for dW accumulation in backward pass.
src file: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py
"""

import math

import torch
import triton
import triton.language as tl

from utils import calculate_settings
from utils import ensure_contiguous

from triton.language.extra.libdevice import rsqrt

@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = x_i / (RMS) * W_i, RMS = sqrt(sum(x_i^2) / N)

    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr, rstd)

    # load W
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    Y_row = X_row * rstd * W_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    dW_ptr,
    X_ptr,
    X_row_stride,
    RSTD_ptr,
    W_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w - (1 / N) * (1 / RMS^2) * ((dy * w) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dW = sum(dy * (x / RMS)). summation over BxT dimension
    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_ptr += row_idx * dY_row_stride
    dX_ptr += row_idx * dX_row_stride

    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx

    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Get cached rms
    rstd_row = tl.load(RSTD_ptr)

    dX_row = rstd_row * (dY_row * W_row)

    dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(dY_row * W_row * X_row, axis=0) * X_row)

    tl.store(dX_ptr + col_offsets, dX_row, mask=mask)

    dW_row = dY_row * (X_row * rstd_row)
    tl.atomic_add(dW_ptr + col_offsets, dW_row.to(tl.float32), mask=mask)


def rms_norm_forward(X, W, eps):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is to cache rstd for each row
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    _rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        RSTD,
        RSTD.stride(0),
        W,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps


def rms_norm_backward(dY, X, RSTD, W, BLOCK_SIZE, num_warps):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    dX = torch.zeros_like(dY)
    dW = torch.zeros_like(W, dtype=torch.float32, device=W.device)

    _rms_norm_backward_kernel[(n_rows,)](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        dW,
        X,
        X.stride(0),
        RSTD,
        W,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dX = dX.view(*shape)

    return dX, dW.to(W.dtype)


class LigerRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps=1e-6):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        Y, X, RSTD, BLOCK_SIZE, num_warps = rms_norm_forward(X, W, eps)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, RSTD, W)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        X, RSTD, W = ctx.saved_tensors
        dX, dW = rms_norm_backward(
            dY, X, RSTD, W, ctx.BLOCK_SIZE, ctx.num_warps,
        )
        return dX, dW, None


def rms_norm(x, weight, eps=1e-6):
    return LigerRMSNormFunction.apply(x, weight, eps)
