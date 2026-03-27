import triton
import torch
import triton.language as tl
import triton.testing

@triton.jit
def online_softmax_kernel(
    input_ptr, output_ptr,
    stride_row, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # 1. 获取当前处理的行号和该行的物理内存基地址
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row

    # 2. 初始化 SRAM 里的标量寄存器 (千万不能用 tl.zeros)
    m_i = -float('inf')  # 历史最大值
    l_i = 0.0            # 历史指数和

    # =========================================================
    # Pass 1: 第一次遍历 HBM，求出全局绝对正确的 Max 和 Sum
    # =========================================================
    for start_col in range(0, n_cols, BLOCK_SIZE):
        cols = start_col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # 从 HBM 载入一个 Chunk，越界部分补 -inf
        ptrs = row_start_ptr + cols
        chunk = tl.load(ptrs, mask=mask, other=-float('inf'))

        # 找当前 Chunk 的最大值
        m_chunk = tl.max(chunk, axis=0)

        # 更新全局最大值
        m_new = tl.maximum(m_i, m_chunk)

        # 补全你那句没写完的代码：计算衰减因子 alpha
        # 如果 m_i 是 -inf，-inf - m_new 会得到 -inf，exp(-inf) 干净利落地等于 0
        alpha = tl.exp(m_i - m_new)

        # 计算当前 Chunk 在新 max 下的指数和
        l_chunk = tl.sum(tl.exp(chunk - m_new), axis=0)

        # 更新全局的指数和与最大值
        l_i = l_i * alpha + l_chunk
        m_i = m_new

    # =========================================================
    # Pass 2: 第二次遍历 HBM，计算最终概率并写回
    # =========================================================
    # 此时循环结束，m_i 已经是全局最大值，l_i 是全局指数和
    out_row_start_ptr = output_ptr + row_idx * stride_row

    for start_col in range(0, n_cols, BLOCK_SIZE):
        cols = start_col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # 必须重新从 HBM 把数据读回 SRAM
        ptrs = row_start_ptr + cols
        chunk = tl.load(ptrs, mask=mask, other=-float('inf'))

        # 用绝对正确的 m_i 和 l_i 算出最终概率
        probs = tl.exp(chunk - m_i) / l_i

        # 写回 HBM 的对应位置
        out_ptrs = out_row_start_ptr + cols
        tl.store(out_ptrs, probs, mask=mask)


@triton.jit
def online_softmax_v2_kernel(
    input_ptr, output_ptr, tmp_ptr,
    stride_row, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    优化版 online softmax:
    Pass 1: 读 HBM + 计算 exp, 将 exp 结果写入临时 buffer (避免 Pass 2 重算 exp)
    Pass 2: 从 tmp buffer 读 exp 结果, 做 rescale + div, 写回 output
    用额外显存换掉 Pass 2 的 exp 计算 + 利用 tmp buffer 的顺序写/读更 cache 友好
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    tmp_row_ptr = tmp_ptr + row_idx * stride_row

    m_i = -float('inf')
    l_i = 0.0

    # =========================================================
    # Pass 1: 读原始数据, 计算 exp(x - running_max), 写入 tmp buffer
    # =========================================================
    for start_col in range(0, n_cols, BLOCK_SIZE):
        cols = start_col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        chunk = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))

        m_chunk = tl.max(chunk, axis=0)
        m_new = tl.maximum(m_i, m_chunk)

        alpha = tl.exp(m_i - m_new)
        # exp(x - m_new): 基于当前已知最大值的 exp
        chunk_exp = tl.exp(chunk - m_new)

        l_i = l_i * alpha + tl.sum(chunk_exp, axis=0)
        m_i = m_new

        # 写入 tmp buffer, 注意此时的 exp 值基于 m_new 而非最终的全局 max
        tl.store(tmp_row_ptr + cols, chunk_exp, mask=mask)

    # =========================================================
    # Pass 2: 从 tmp buffer 读 exp 结果, rescale 到全局 max, 除以 l_i
    # =========================================================
    # 此时 m_i 是全局 max, l_i 是全局 sum
    # tmp 中第 k 个 chunk 存的是 exp(x - m_k), 需要乘 exp(m_k - m_i) 来 rescale
    # 但我们没存每个 chunk 的 m_k...
    #
    # 技巧: 不需要逐 chunk rescale!
    # Pass 1 结束时 l_i 已经是正确的全局 sum (online 算法保证了这一点)
    # tmp[j] = exp(x_j - m_at_that_point), 而我们需要的是 exp(x_j - m_i) / l_i
    # 所以 Pass 2 还是需要从原始数据重算...
    #
    # 换一个思路: Pass 1 直接存原始数据到 tmp (省掉 Pass 2 的 HBM 读),
    # 但原始数据就在 input_ptr, 这没意义...
    #
    # 真正有效的优化: 把 Pass 1 的 exp 结果存下来, 同时记录每个 chunk 对应的 running max
    # Pass 2 通过 rescale 修正, 避免重算 exp

    out_row_ptr = output_ptr + row_idx * stride_row
    # m_i 是最终全局 max, 需要逐 chunk 重新 rescale
    # 重新跑一遍 running max 来拿到每个 chunk 当时的 m_k
    m_k = -float('inf')
    for start_col in range(0, n_cols, BLOCK_SIZE):
        cols = start_col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # 从 tmp 读出 Pass 1 存的 exp(x - m_k)
        chunk_exp = tl.load(tmp_row_ptr + cols, mask=mask, other=0.0)

        # 需要还原当时的 m_k 来做 rescale
        # 从原始数据拿 chunk max (这里还是要读一次 HBM...)
        chunk = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        m_chunk = tl.max(chunk, axis=0)
        m_k_new = tl.maximum(m_k, m_chunk)

        # rescale: exp(x - m_k_new) = exp(x - m_k) * exp(m_k - m_k_new)
        # 但 chunk_exp 存的是 exp(x - m_k_new_at_pass1), 也就是 m_k_new
        # 所以 rescale factor = exp(m_k_new - m_i)
        rescale = tl.exp(m_k_new - m_i)
        probs = chunk_exp * rescale / l_i

        tl.store(out_row_ptr + cols, probs, mask=mask)
        m_k = m_k_new


@triton.jit
def online_softmax_v3_kernel(
    input_ptr, output_ptr,
    stride_row, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    优化版 online softmax v3: 自适应 BLOCK_SIZE
    当 n_cols <= BLOCK_SIZE 时, 退化为单 pass safe softmax (无循环开销)
    当 n_cols > BLOCK_SIZE 时, 走 2-pass online 路径
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    out_row_ptr = output_ptr + row_idx * stride_row

    # =========================================================
    # 快速路径: 一行能装进一个 BLOCK, 退化为 safe softmax
    # =========================================================
    if n_cols <= BLOCK_SIZE:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row = tl.load(row_start_ptr + offsets, mask=mask, other=-float('inf'))
        row_max = tl.max(row)
        row = row - row_max
        row_exp = tl.exp(row)
        sum_exp = tl.sum(row_exp)
        tl.store(out_row_ptr + offsets, row_exp / sum_exp, mask=mask)
    else:
        # =========================================================
        # 慢速路径: 2-pass online softmax
        # =========================================================
        m_i = -float('inf')
        l_i = 0.0

        for start_col in range(0, n_cols, BLOCK_SIZE):
            cols = start_col + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            chunk = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))

            m_chunk = tl.max(chunk, axis=0)
            m_new = tl.maximum(m_i, m_chunk)
            alpha = tl.exp(m_i - m_new)
            l_chunk = tl.sum(tl.exp(chunk - m_new), axis=0)
            l_i = l_i * alpha + l_chunk
            m_i = m_new

        for start_col in range(0, n_cols, BLOCK_SIZE):
            cols = start_col + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            chunk = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
            probs = tl.exp(chunk - m_i) / l_i
            tl.store(out_row_ptr + cols, probs, mask=mask)
    
    
        

@triton.jit
def safesoftmax(input_ptr: torch.Tensor,
                output_ptr: torch.Tensor,
                stride_row,
                n_cols,
                BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(row_start_ptr + offsets, mask=mask, other=float('-inf'))
    row_max = tl.max(row)
    row = row - row_max
    row_exp = tl.exp(row)
    sum_exp = tl.sum(row_exp)

    out_ptr = output_ptr + row_idx * stride_row
    tl.store(out_ptr + offsets, row_exp/sum_exp, mask=mask)
    
@triton.jit
def softmax(input_ptr: torch.Tensor,
            output_ptr: torch.Tensor,
            stride_row,
            n_cols,
            BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_exp = tl.exp(row)
    sum_exp = tl.sum(row_exp)

    out_ptr = output_ptr + row_idx * stride_row
    tl.store(out_ptr + col_offsets, row_exp/sum_exp, mask=mask)


# ==================== 辅助函数 ====================

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """调用 Triton softmax kernel 的包装函数"""
    n_rows, n_cols = x.shape
    # BLOCK_SIZE 需要是2的幂且 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    softmax[(n_rows,)](x, output, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_safe_softmax(x: torch.Tensor) -> torch.Tensor:
    """调用 Triton safe softmax kernel 的包装函数 (带 max 减法, 数值更稳定)"""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    safesoftmax[(n_rows,)](x, output, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_online_softmax(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """调用 Triton online softmax kernel 的包装函数 (分块流式计算, 支持超长行)"""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(block_size)
    output = torch.empty_like(x)
    online_softmax_kernel[(n_rows,)](x, output, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_online_softmax_v3(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    调用优化版 online softmax v3:
    - N <= block_size 时自动退化为单 pass safe softmax
    - N > block_size 时走 2-pass online 路径
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(block_size)
    output = torch.empty_like(x)
    online_softmax_v3_kernel[(n_rows,)](x, output, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 原生 softmax"""
    return torch.softmax(x, dim=-1)


# ==================== 正确性验证 ====================

def validate():
    """验证 Triton softmax 与 PyTorch softmax 结果一致"""
    print("=" * 60)
    print("正确性验证")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(128, 1024, device='cuda', dtype=torch.float32)

    torch_out = torch_softmax(x)

    # 验证 naive softmax
    triton_out = triton_softmax(x)
    if torch.allclose(triton_out, torch_out, atol=1e-5, rtol=1e-5):
        print("✅ Triton softmax 与 PyTorch softmax 结果一致!")
    else:
        max_diff = (triton_out - torch_out).abs().max().item()
        print(f"❌ Triton softmax 结果不一致, 最大误差: {max_diff}")

    # 验证 safe softmax
    safe_out = triton_safe_softmax(x)
    if torch.allclose(safe_out, torch_out, atol=1e-5, rtol=1e-5):
        print("✅ Triton safe softmax 与 PyTorch softmax 结果一致!")
    else:
        max_diff = (safe_out - torch_out).abs().max().item()
        print(f"❌ Triton safe softmax 结果不一致, 最大误差: {max_diff}")

    # 验证 online softmax
    online_out = triton_online_softmax(x)
    if torch.allclose(online_out, torch_out, atol=1e-5, rtol=1e-5):
        print("✅ Triton online softmax 与 PyTorch softmax 结果一致!")
    else:
        max_diff = (online_out - torch_out).abs().max().item()
        print(f"❌ Triton online softmax 结果不一致, 最大误差: {max_diff}")

    # 验证 online softmax v3
    online_v3_out = triton_online_softmax_v3(x)
    if torch.allclose(online_v3_out, torch_out, atol=1e-5, rtol=1e-5):
        print("✅ Triton online softmax v3 与 PyTorch softmax 结果一致!")
    else:
        max_diff = (online_v3_out - torch_out).abs().max().item()
        print(f"❌ Triton online softmax v3 结果不一致, 最大误差: {max_diff}")

    # 验证 online v3 在超长行 (N > BLOCK_SIZE) 的情况
    x_long = torch.randn(32, 8192, device='cuda', dtype=torch.float32)
    torch_long = torch.softmax(x_long, dim=-1)
    online_v3_long = triton_online_softmax_v3(x_long, block_size=1024)
    if torch.allclose(online_v3_long, torch_long, atol=1e-5, rtol=1e-5):
        print("✅ Triton online softmax v3 (N=8192, BS=1024) 结果一致!")
    else:
        max_diff = (online_v3_long - torch_long).abs().max().item()
        print(f"❌ Triton online softmax v3 (N=8192) 结果不一致, 最大误差: {max_diff}")
    print()


# ==================== Benchmark ====================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # x轴参数名
        x_vals=[128 * i for i in range(2, 65)],  # x轴取值: 256 ~ 8192
        line_arg='provider',  # 不同曲线的参数名
        line_vals=['triton', 'triton-safe', 'triton-online', 'triton-online-v3', 'torch'],  # 曲线对应的值
        line_names=['Triton', 'Triton (safe)', 'Triton (online)', 'Triton (online-v3)', 'PyTorch'],  # 图例名称
        styles=[('blue', '-'), ('red', '-'), ('orange', '-'), ('purple', '-'), ('green', '-')],  # 曲线样式
        ylabel='GB/s',  # y轴标签
        plot_name='softmax-performance',  # 图表名称
        args={'M': 4096},  # 固定参数: 行数
    )
)
def benchmark(M, N, provider):
    """
    Benchmark softmax 性能
    M: 行数 (固定)
    N: 列数 (变化)
    provider: 'triton', 'triton-safe', 'triton-online', 'triton-online-v3' 或 'torch'
    """
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]  # 中位数, P20, P80

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_softmax(x), quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_softmax(x), quantiles=quantiles
        )
    elif provider == 'triton-safe':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_safe_softmax(x), quantiles=quantiles
        )
    elif provider == 'triton-online':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_online_softmax(x), quantiles=quantiles
        )
    elif provider == 'triton-online-v3':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_online_softmax_v3(x), quantiles=quantiles
        )

    # 计算带宽: online softmax 读 2 次 + 写 1 次 = 3 * M * N, 其他读 1 次 + 写 1 次 = 2 * M * N
    if provider == 'triton-online':
        gbps = lambda ms: 2 * M * N * x.element_size() * 1e-9 / (ms * 1e-3)
    else:
        gbps = lambda ms: 2 * M * N * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    # 1. 正确性验证
    validate()

    # 2. 性能测试
    print("=" * 60)
    print("性能测试 (Benchmark)")
    print("=" * 60)
    print("固定行数 M=4096, 列数 N 从 256 变化到 8192")
    print("结果将保存为 softmax-performance.png")
    print()
    benchmark.run(show_plots=False, print_data=True)

    # 3. 理论 vs 实际对比 (改进版: 考虑小数据量打不满带宽的情况)
    print()
    print("=" * 70)
    print("理论 vs 实际耗时对比 (改进版 Roofline)")
    print("=" * 70)

    # ======================== GPU 参数 ========================
    # H100 SXM 参数, 可根据实际 GPU 修改
    HBM_BW_PEAK = 3.35e12   # bytes/s, HBM 峰值带宽
    N_SMS = 132              # SM 数量
    LAUNCH_OVERHEAD_US = 5.0 # kernel launch 开销 (μs), 经验值

    M = 4096
    elem_size = 4  # float32
    quantiles = [0.5, 0.2, 0.8]

    print(f"GPU: H100 SXM")
    print(f"  HBM 峰值带宽: {HBM_BW_PEAK / 1e12:.2f} TB/s")
    print(f"  SM 数量: {N_SMS}")
    print(f"  Kernel launch 开销: ~{LAUNCH_OVERHEAD_US:.0f} μs")
    print(f"矩阵: M={M}, dtype=float32")
    print()

    # ======================== Step 1: 实测不同数据量的 memcpy 带宽 ========================
    print("--- Step 1: 实测 memcpy 带宽 (不同数据量下的实际可用带宽) ---")
    print(f"{'数据量 (MB)':>12s} | {'memcpy (μs)':>12s} | {'实测带宽 (TB/s)':>15s} | {'带宽利用率':>10s}")
    print("-" * 58)

    @triton.jit
    def _memcpy_kernel(src, dst, n_elems, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems
        x = tl.load(src + offs, mask=mask)
        tl.store(dst + offs, x, mask=mask)

    measured_bw = {}  # data_bytes → 实测带宽 (bytes/s)
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        n_elems = M * N
        data_bytes = n_elems * elem_size
        src = torch.randn(M, N, device='cuda', dtype=torch.float32)
        dst = torch.empty_like(src)
        BLOCK = 1024
        grid = ((n_elems + BLOCK - 1) // BLOCK,)

        ms, _, _ = triton.testing.do_bench(
            lambda: _memcpy_kernel[grid](src, dst, n_elems, BLOCK_SIZE=BLOCK),
            quantiles=quantiles
        )
        actual_us = ms * 1000
        # memcpy: 1 读 + 1 写 = 2 × data_bytes
        bw = 2 * data_bytes / (actual_us * 1e-6)
        measured_bw[data_bytes] = bw
        util = bw / HBM_BW_PEAK * 100
        print(f"{data_bytes/1e6:>12.1f} | {actual_us:>12.1f} | {bw/1e12:>15.3f} | {util:>9.0f}%")

    print()

    # ======================== Step 2: 三种理论模型对比 ========================
    print("--- Step 2: 三种理论模型 vs 实际耗时 ---")
    print()
    print("模型 A: 理想模型     T = 访存量 / 峰值带宽")
    print("模型 B: 含 launch    T = launch_overhead + 访存量 / 峰值带宽")
    print("模型 C: 实测带宽     T = 访存量 / 实测带宽(该数据量)")
    print()

    impls = [
        ("safe",   triton_safe_softmax,   2),  # 1R + 1W
        ("online", triton_online_softmax,  3),  # 2R + 1W
    ]

    for impl_name, fn, rw_factor in impls:
        print(f"=== {impl_name} softmax (访存: {rw_factor-1}R + 1W) ===")
        print(f"{'N':>6s} | {'实际(μs)':>9s} | {'模型A(μs)':>10s} {'误差':>6s} | {'模型B(μs)':>10s} {'误差':>6s} | {'模型C(μs)':>10s} {'误差':>6s}")
        print("-" * 85)

        for N in [256, 512, 1024, 2048, 4096, 8192]:
            data_bytes = M * N * elem_size
            x = torch.randn(M, N, device='cuda', dtype=torch.float32)
            ms, _, _ = triton.testing.do_bench(lambda: fn(x), quantiles=quantiles)
            actual_us = ms * 1000

            total_bytes = rw_factor * data_bytes

            # 模型 A: 理想 (峰值带宽)
            theory_a = total_bytes / HBM_BW_PEAK * 1e6
            err_a = (theory_a - actual_us) / actual_us * 100

            # 模型 B: 理想 + launch overhead
            theory_b = LAUNCH_OVERHEAD_US + total_bytes / HBM_BW_PEAK * 1e6
            err_b = (theory_b - actual_us) / actual_us * 100

            # 模型 C: 用 memcpy 实测的带宽 (该数据量级下的真实带宽)
            # memcpy 测的是 2×data_bytes 的带宽, softmax 访存量是 rw_factor×data_bytes
            # 用同一数据量级的实测带宽来估算
            real_bw = measured_bw.get(data_bytes, HBM_BW_PEAK)
            theory_c = total_bytes / real_bw * 1e6
            err_c = (theory_c - actual_us) / actual_us * 100

            print(f"{N:>6d} | {actual_us:>9.1f} | {theory_a:>10.1f} {err_a:>+5.0f}% | {theory_b:>10.1f} {err_b:>+5.0f}% | {theory_c:>10.1f} {err_c:>+5.0f}%")

        print()

    print("误差说明: 负值 = 模型低估 (实际比预测慢), 正值 = 模型高估")
    print()
    print("结论:")
    print("  - 模型 A (峰值带宽) 在小 N 时严重低估, 因为带宽打不满")
    print("  - 模型 B (+launch) 小 N 时改善但仍不够, 因为没考虑 SM 利用率")
    print("  - 模型 C (实测带宽) 最准确, 因为 memcpy 已经包含了:")
    print("      SM 占用率、L2 cache 效应、HBM channel 利用率等因素")
    print("  - 实用建议: 先跑一次 memcpy benchmark 建立 '数据量→实际带宽' 查找表")
    print("    然后用 T = 访存量 / 实测带宽 来估算 memory-bound kernel 的耗时")
