import functools

import torch
import triton


def is_hip():
    """Check if running on AMD ROCm (HIP) platform."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def calculate_settings(n):
    """
    Calculate optimal Triton kernel launch settings (BLOCK_SIZE, num_warps)
    based on the hidden dimension size.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/utils.py
    """
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return BLOCK_SIZE, num_warps


def ensure_contiguous(fn):
    """
    Decorator that ensures all torch.Tensor arguments are contiguous
    before passing them to the wrapped autograd Function method.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper
