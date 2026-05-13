"""
Step 2: Triton Softmax Kernel 实现

运行方式: python softmax_kernel.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐行计算 softmax

    每个 program instance 处理一行。

    关键点:
    1. 使用 tl.max 找最大值 (数值稳定性)
    2. 使用 tl.sum 计算归一化因子
    3. mask 处理不满 BLOCK_SIZE 的情况
    """
    # 1. 获取当前处理的行号
    row_idx = tl.program_id(0)

    # 2. 计算当前行的起始地址
    row_start = row_idx * input_row_stride

    # 3. 计算列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 4. 构造 mask (处理不满 BLOCK_SIZE 的情况)
    mask = col_offsets < n_cols

    # 5. 加载一行数据
    # other=-float('inf') 确保 mask=False 的位置不影响 max 计算
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # ========== 核心 Softmax 计算 ==========

    # Step 1: 找最大值 (数值稳定性)
    # 如果不减去 max, exp 可能溢出
    row_max = tl.max(row, axis=0)

    # Step 2: 计算 exp(x - max)
    # 减去 max 确保 exp 的参数 <= 0, 不会溢出
    numerator = tl.exp(row - row_max)

    # Step 3: 计算分母 (sum of exp)
    denominator = tl.sum(numerator, axis=0)

    # Step 4: 归一化
    softmax_output = numerator / denominator

    # ========================================

    # 6. 写回结果
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Host 函数: 调用 softmax kernel

    Args:
        x: 输入张量, shape [n_rows, n_cols]
    Returns:
        softmax(x), shape [n_rows, n_cols]
    """
    assert x.is_cuda, "输入必须在 GPU 上"
    assert x.dim() == 2, "输入必须是 2D 张量"

    n_rows, n_cols = x.shape

    # 分配输出
    output = torch.empty_like(x)

    # 选择 BLOCK_SIZE: 大于 n_cols 的最小 2 的幂
    # 这样可以最大化内存合并访问
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 启动 kernel
    # grid: (n_rows,) 表示启动 n_rows 个 program instances
    # 每个 program instance 处理一行
    grid = (n_rows,)

    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LogSoftmax kernel 练习

    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    shifted = row - row_max
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)

    # log_softmax = x - max - log(sum(exp(x - max)))
    log_softmax_output = shifted - tl.log(denominator)

    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, log_softmax_output, mask=mask)


def triton_log_softmax(x: torch.Tensor):
    """LogSoftmax host 函数"""
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    log_softmax_kernel[grid](
        output, x, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def test_correctness():
    """测试正确性"""
    print("=" * 60)
    print("测试 Softmax Kernel 正确性")
    print("=" * 60)

    torch.manual_seed(42)

    # 测试不同大小
    test_cases = [
        (4, 16),
        (16, 64),
        (32, 128),
        (8, 1024),
    ]

    for n_rows, n_cols in test_cases:
        x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)

        # Triton 结果
        triton_out = triton_softmax(x)

        # PyTorch 参考结果
        torch_out = F.softmax(x, dim=1)

        # 检查正确性
        diff = (triton_out - torch_out).abs().max().item()

        status = "✓ PASS" if diff < 1e-5 else "✗ FAIL"
        print(f"  Shape ({n_rows:4d}, {n_cols:4d}): max_diff = {diff:.2e} {status}")

    print()


def test_log_softmax():
    """测试 LogSoftmax"""
    print("=" * 60)
    print("测试 LogSoftmax Kernel 正确性")
    print("=" * 60)

    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)

    triton_out = triton_log_softmax(x)
    torch_out = F.log_softmax(x, dim=1)

    diff = (triton_out - torch_out).abs().max().item()
    status = "✓ PASS" if diff < 1e-5 else "✗ FAIL"
    print(f"  Shape (16, 128): max_diff = {diff:.2e} {status}")
    print()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 参数名
        x_vals=[128, 256, 512, 1024, 2048, 4096],  # 参数值
        line_arg='provider',  # 图例中的分类
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Time (ms)',
        plot_name='softmax-performance',
        args={'M': 4096},  # 固定行数
    )
)
def benchmark(M, N, provider):
    """性能基准测试"""
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.softmax(x, dim=1), quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x), quantiles=quantiles
        )

    return ms, max_ms, min_ms


def main():
    test_correctness()
    test_log_softmax()

    print("=" * 60)
    print("性能基准测试 (M=4096 行)")
    print("=" * 60)

    # 运行 benchmark
    benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()