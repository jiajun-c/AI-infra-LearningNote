"""
分析 safe softmax 为什么比 naive softmax 更快
已排除: exp 本身对不同值域的性能无差异

本脚本从多个角度定位真正原因:
1. 逐步拆解 softmax 各阶段的耗时
2. 对比两个 kernel 编译后的指令数 (PTX/TTGIR)
3. 测试 sum_exp 的值域对除法性能的影响
"""
import triton
import torch
import triton.language as tl
import triton.testing


# ==================== Kernel 变体: 拆解各阶段 ====================

@triton.jit
def kernel_load_only(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """只做 load + store"""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + row_idx * n_cols + offsets, row, mask=mask)


@triton.jit
def kernel_exp_only(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """load + exp + store"""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=0.0)
    result = tl.exp(row)
    tl.store(output_ptr + row_idx * n_cols + offsets, result, mask=mask)


@triton.jit
def kernel_exp_sum(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """load + exp + sum + store"""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=0.0)
    row_exp = tl.exp(row)
    sum_exp = tl.sum(row_exp)
    # store sum_exp 到第一个位置, 防止被优化掉
    tl.store(output_ptr + row_idx * n_cols + offsets, row_exp + sum_exp, mask=mask)


@triton.jit
def kernel_naive_softmax(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """完整 naive softmax: load + exp + sum + div + store"""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    row_exp = tl.exp(row)
    sum_exp = tl.sum(row_exp)
    tl.store(output_ptr + row_idx * n_cols + offsets, row_exp / sum_exp, mask=mask)


@triton.jit
def kernel_safe_softmax(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """完整 safe softmax: load + max + sub + exp + sum + div + store"""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row)
    row = row - row_max
    row_exp = tl.exp(row)
    sum_exp = tl.sum(row_exp)
    tl.store(output_ptr + row_idx * n_cols + offsets, row_exp / sum_exp, mask=mask)


# ==================== 包装函数 ====================

def make_runner(kernel_fn):
    def run(x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        output = torch.empty_like(x)
        kernel_fn[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output
    return run


# ==================== 测试1: 逐步拆解各阶段耗时 ====================

def bench_breakdown():
    """拆解 softmax 各阶段, 看耗时增量在哪"""
    print("=" * 70)
    print("测试1: softmax 各阶段耗时拆解 (4096 x 8192)")
    print("=" * 70)
    print(f"{'阶段':>25s} | {'耗时 (ms)':>10s} | {'增量 (ms)':>10s} | {'GB/s':>10s}")
    print("-" * 65)

    M, N = 4096, 8192
    quantiles = [0.5, 0.2, 0.8]
    x = torch.empty(M, N, device='cuda', dtype=torch.float32).uniform_(-10.0, 10.0)

    kernels = [
        ("load + store",             kernel_load_only),
        ("load + exp + store",       kernel_exp_only),
        ("load + exp + sum + store", kernel_exp_sum),
        ("naive softmax (完整)",     kernel_naive_softmax),
        ("safe softmax (完整)",      kernel_safe_softmax),
    ]

    prev_ms = 0.0
    for name, kernel_fn in kernels:
        runner = make_runner(kernel_fn)
        ms, _, _ = triton.testing.do_bench(lambda: runner(x), quantiles=quantiles)
        gbps = 2 * M * N * x.element_size() * 1e-9 / (ms * 1e-3)
        delta = ms - prev_ms
        print(f"{name:>25s} | {ms:>10.3f} | {delta:>+10.3f} | {gbps:>10.1f}")
        prev_ms = ms

    print()


# ==================== 测试2: 变化 N, naive vs safe 对比 ====================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 65)],
        line_arg='provider',
        line_vals=['naive', 'safe'],
        line_names=['naive softmax', 'safe softmax'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='softmax-naive-vs-safe',
        args={'M': 4096},
    )
)
def bench_by_size(M, N, provider):
    """变化 N, 对比 naive vs safe"""
    quantiles = [0.5, 0.2, 0.8]
    x = torch.empty(M, N, device='cuda', dtype=torch.float32).uniform_(-10.0, 10.0)

    if provider == 'naive':
        runner = make_runner(kernel_naive_softmax)
    elif provider == 'safe':
        runner = make_runner(kernel_safe_softmax)

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: runner(x), quantiles=quantiles)

    gbps = lambda ms: 2 * M * N * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ==================== 测试3: 打印两个 kernel 的编译信息 ====================

def print_kernel_info():
    """打印两个 kernel 编译后的寄存器/shared memory 使用量"""
    print("=" * 70)
    print("测试3: Kernel 编译信息对比")
    print("=" * 70)

    M, N = 4096, 8192
    BLOCK_SIZE = triton.next_power_of_2(N)
    x = torch.empty(M, N, device='cuda', dtype=torch.float32).uniform_(-10.0, 10.0)
    output = torch.empty_like(x)

    for name, kernel_fn in [("naive softmax", kernel_naive_softmax),
                             ("safe softmax", kernel_safe_softmax)]:
        # 触发编译
        kernel_fn[(M,)](x, output, N, BLOCK_SIZE=BLOCK_SIZE)

        # 尝试获取编译信息
        try:
            compiled = kernel_fn.cache[0]
            if compiled:
                key = list(compiled.keys())[0]
                info = compiled[key]
                print(f"\n{name}:")
                print(f"  n_regs    = {info.n_regs}")
                print(f"  n_spills  = {info.n_spills}")
                print(f"  shared    = {info.metadata.shared} bytes")
        except Exception as e:
            print(f"\n{name}: 无法获取编译信息 ({e})")

    print()


if __name__ == '__main__':
    # 测试1: 拆解各阶段
    bench_breakdown()

    # 测试3: 编译信息
    print_kernel_info()

    # 测试2: 变化 N
    print("=" * 70)
    print("测试2: 变化 N, naive vs safe softmax 带宽对比")
    print("=" * 70)
    print("结果将保存为 softmax-naive-vs-safe.png")
    print()
    bench_by_size.run(show_plots=False, print_data=True)
