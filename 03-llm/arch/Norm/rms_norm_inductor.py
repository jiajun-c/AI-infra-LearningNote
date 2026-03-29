"""
RMS Norm Inductor 风格实现 —— 从 torch.compile 生成的代码中提炼的手写 Triton kernel。

对比三种 backward dW 累积策略:
  1. Atomic:  每行一个 program, tl.atomic_add 到全局 dW           (争用严重)
  2. SM:      每 SM 一个 program, 循环处理多行, 本地累加后写出       (并行度低)
  3. Inductor: 两级 parallel reduce, 高并行 + GPU 端全量 reduce   (最优)

核心思想:
  - Forward:  单 kernel, 两趟循环 (reduce 算 rstd + elementwise 算 Y)
  - Backward: 3 个 kernel 流水线
    * Kernel 1: dW 部分累加 (高并行, grid=n_cols×n_chunks)
    * Kernel 2: dW 最终 reduce (grid=n_cols)
    * Kernel 3: dX 计算 (每行独立, 两趟循环)
  - dX 和 dW 计算完全解耦, 可以 overlap 执行

Reference: torch.compile + inductor 对 torch.nn.RMSNorm 的自动优化输出
"""

import math

import torch
import triton
import triton.language as tl

from utils import calculate_settings
from utils import ensure_contiguous

from triton.language.extra.libdevice import rsqrt


# ===========================================================================
# Forward Kernel (与 SM 版本相同, inductor 生成的结构也一样)
# ===========================================================================

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
    Forward: Y = X * rsqrt(mean(X^2) + eps) * W

    Grid: (n_rows,)  每行一个 program
    两趟循环:
      趟 1: reduce sum(X^2) → 算 rstd
      趟 2: elementwise Y = X * rstd * W
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    # ----- 趟 1: reduce -----
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # 缓存 rstd 供 backward 使用 (只有 n_rows 个标量, 开销极小)
    tl.store(RSTD_ptr, rstd)

    # ----- 趟 2: elementwise -----
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    Y_row = X_row * rstd * W_row
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


# ===========================================================================
# Backward Kernel 1: dW 部分累加 (OUTER reduction)
# ===========================================================================
#
# 关键思想: 将 [n_rows, n_cols] 的 dW 贡献矩阵视为二维 reduce 问题
#
#   dW[col] = Σ_{row} dY[row, col] * X[row, col] * rstd[row]
#
# 朴素做法: 沿 row 维度 reduce → 但 n_rows 可能很大 (128K), 一次 reduce 很慢
#
# Inductor 做法: 两级 reduce
#   第 1 级: 把 n_rows 切成 n_chunks 块, 每块 ROWS_PER_CHUNK 行
#            grid = (n_cols_blocks × n_chunks), 每个 program 处理 XBLOCK 列 × ROWS_PER_CHUNK 行
#            输出中间结果 buf[n_chunks, n_cols]
#   第 2 级: 对 buf 沿 n_chunks 维度 reduce → 最终 dW[n_cols]
#
# 为什么比 SM 方案快?
#   SM 方案: grid = sm_count ≈ 132, 每个 program 循环 ~993 行
#   Inductor: grid = (n_cols/XBLOCK) × n_chunks = 大量 programs
#             并行度高, GPU occupancy 饱满
#
# 访存优化:
#   每个 program 处理 XBLOCK 个连续列 × RBLOCK 行,
#   dY/X 按行主序存储, 同一 warp 内的 thread 访问连续列 → coalesced access

@triton.jit
def _dw_partial_reduce_kernel(
    dY_ptr,        # [n_rows, n_cols]  上游梯度
    X_ptr,         # [n_rows, n_cols]  输入
    RSTD_ptr,      # [n_rows]          缓存的 1/RMS
    dW_buf_ptr,    # [n_chunks, n_cols] 中间输出
    n_rows,
    n_cols,
    stride_dy_row,
    stride_x_row,
    stride_buf_row,
    n_chunks,
    ROWS_PER_CHUNK: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """
    Backward Kernel 1: dW 的部分累加

    Grid: (n_cols // XBLOCK × n_chunks,)
    每个 program 负责:
      - XBLOCK 个连续列 (coalesced access)
      - ROWS_PER_CHUNK 行 (loop + reduce)
    输出: dW_buf[chunk_id, col_block_start : col_block_start + XBLOCK]
    """
    pid = tl.program_id(0)
    n_col_blocks = (n_cols + XBLOCK - 1) // XBLOCK
    # 解码: col_block 在低位 (保证相邻 pid 处理相邻列), chunk 在高位
    col_block_id = pid % n_col_blocks
    chunk_id = pid // n_col_blocks

    # 本 program 负责的列范围
    col_offsets = col_block_id * XBLOCK + tl.arange(0, XBLOCK)
    col_mask = col_offsets < n_cols

    # 本 chunk 负责的行范围
    row_start = chunk_id * ROWS_PER_CHUNK
    row_end = min(row_start + ROWS_PER_CHUNK, n_rows)

    # 累加器: XBLOCK 列的局部和 (寄存器, 零争用)
    acc = tl.zeros([XBLOCK], dtype=tl.float32)

    # 遍历本 chunk 内的所有行, 每次处理 RBLOCK 行
    for row_offset in range(0, ROWS_PER_CHUNK, RBLOCK):
        row_ids = row_start + row_offset + tl.arange(0, RBLOCK)
        row_mask = row_ids < row_end

        # [RBLOCK, XBLOCK] 的 2D load — 行主序, warp 内连续列 → coalesced
        dy_vals = tl.load(
            dY_ptr + row_ids[:, None] * stride_dy_row + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :], other=0.0
        ).to(tl.float32)

        x_vals = tl.load(
            X_ptr + row_ids[:, None] * stride_x_row + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :], other=0.0
        ).to(tl.float32)

        rstd_vals = tl.load(
            RSTD_ptr + row_ids,
            mask=row_mask, other=0.0
        ).to(tl.float32)

        # dW 贡献 = dY * X * rstd, 沿 row 维度 reduce
        # [RBLOCK, XBLOCK] * [RBLOCK, 1] → [RBLOCK, XBLOCK] → sum → [XBLOCK]
        contrib = dy_vals * x_vals * rstd_vals[:, None]
        acc += tl.sum(contrib, axis=0)

    # 写出到中间 buffer [n_chunks, n_cols]
    tl.store(
        dW_buf_ptr + chunk_id * stride_buf_row + col_offsets,
        acc, mask=col_mask
    )


# ===========================================================================
# Backward Kernel 2: dW 最终 reduce
# ===========================================================================

@triton.jit
def _dw_final_reduce_kernel(
    dW_buf_ptr,    # [n_chunks, n_cols] 中间结果
    dW_ptr,        # [n_cols]           最终输出
    n_cols,
    n_chunks,
    stride_buf_row,
    RBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """
    Backward Kernel 2: 将 n_chunks 个局部和 reduce 成最终 dW

    Grid: (cdiv(n_cols, XBLOCK),)
    每个 program 负责 XBLOCK 列的 reduce:
      dW[col_block] = Σ_{chunk} dW_buf[chunk, col_block]

    n_chunks 通常只有 64, 用 persistent reduction 一次搞定
    """
    pid = tl.program_id(0)
    col_offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    col_mask = col_offsets < n_cols

    # 加载所有 chunk 的局部和并 reduce
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    chunk_offsets = tl.arange(0, RBLOCK)

    for chunk_offset in range(0, n_chunks, RBLOCK):
        chunk_ids = chunk_offset + chunk_offsets
        chunk_mask = chunk_ids < n_chunks
        # [RBLOCK, XBLOCK]
        partial = tl.load(
            dW_buf_ptr + chunk_ids[:, None] * stride_buf_row + col_offsets[None, :],
            mask=chunk_mask[:, None] & col_mask[None, :], other=0.0
        )
        acc += tl.sum(partial, axis=0)

    tl.store(dW_ptr + col_offsets, acc, mask=col_mask)


# ===========================================================================
# Backward Kernel 3: dX 计算 (每行独立, 两趟循环)
# ===========================================================================

@triton.jit
def _dx_kernel(
    dY_ptr,
    dX_ptr,
    X_ptr,
    RSTD_ptr,
    W_ptr,
    stride_dy_row,
    stride_dx_row,
    stride_x_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward Kernel 3: 计算 dX

    Grid: (n_rows,)
    每个 program 负责一行的 dX:
      dX = rstd * (dY*W) + rstd * (-(1/N) * rstd^2 * dot(dY*W, X) * X)
         = rstd * dY * W - rstd^3 / N * dot(dY*W, X) * X

    两趟循环:
      趟 1: reduce dot = Σ(dY * W * X)
      趟 2: elementwise 计算 dX 并写出
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 加载本行数据
    dY_row = tl.load(dY_ptr + row_idx * stride_dy_row + col_offsets,
                     mask=mask, other=0.0).to(tl.float32)
    X_row = tl.load(X_ptr + row_idx * stride_x_row + col_offsets,
                    mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_idx)

    # ----- 趟 1: reduce dot = Σ(dY * W * X) -----
    dY_W = dY_row * W_row
    dot = tl.sum(dY_W * X_row, axis=0)

    # ----- 趟 2: elementwise dX -----
    # dX = rstd * dY*W + (-0.5) * dot * rstd^3 * (2/N) * X
    #    = rstd * dY*W - (1/N) * dot * rstd^3 * X
    term1 = rstd * dY_W
    term2 = -(1.0 / n_cols) * rstd * rstd * rstd * dot * X_row
    dX_row = term1 + term2

    tl.store(dX_ptr + row_idx * stride_dx_row + col_offsets,
             dX_row, mask=mask)


# ===========================================================================
# Python wrapper
# ===========================================================================

def rms_norm_forward(X, W, eps):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    _rms_norm_forward_kernel[(n_rows,)](
        Y, Y.stride(0),
        X, X.stride(0),
        RSTD, RSTD.stride(0),
        W,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps


def rms_norm_backward(dY, X, RSTD, W, BLOCK_SIZE, num_warps):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # ===== Kernel 1: dW 部分累加 =====
    # 选择 chunk 数量: 目标是让每个 chunk 处理 ~512 行 (inductor 的选择)
    ROWS_PER_CHUNK = 512
    n_chunks = math.ceil(n_rows / ROWS_PER_CHUNK)
    RBLOCK = min(ROWS_PER_CHUNK, 64)    # 内层循环每次处理的行数
    XBLOCK = min(BLOCK_SIZE, 128)        # 每个 program 处理的列数

    # 中间 buffer: [n_chunks, n_cols] — 行主序, 方便 kernel2 reduce
    dW_buf = torch.empty((n_chunks, n_cols), dtype=torch.float32, device=X.device)

    n_col_blocks = math.ceil(n_cols / XBLOCK)
    grid_partial = (n_col_blocks * n_chunks,)
    _dw_partial_reduce_kernel[grid_partial](
        dY, X, RSTD, dW_buf,
        n_rows, n_cols,
        dY.stride(0), X.stride(0), dW_buf.stride(0),
        n_chunks,
        ROWS_PER_CHUNK=ROWS_PER_CHUNK,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
    )

    # ===== Kernel 2: dW 最终 reduce =====
    dW = torch.empty(n_cols, dtype=torch.float32, device=X.device)

    RBLOCK_REDUCE = triton.next_power_of_2(n_chunks)
    grid_reduce = (math.ceil(n_cols / XBLOCK),)
    _dw_final_reduce_kernel[grid_reduce](
        dW_buf, dW,
        n_cols, n_chunks,
        dW_buf.stride(0),
        RBLOCK=RBLOCK_REDUCE,
        XBLOCK=XBLOCK,
    )

    # ===== Kernel 3: dX =====
    dX = torch.empty_like(dY)

    _dx_kernel[(n_rows,)](
        dY, dX, X, RSTD, W,
        dY.stride(0), dX.stride(0), X.stride(0),
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
