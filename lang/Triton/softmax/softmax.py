import triton
import triton.language as tl
import torch

# -----------------------------------------------------------------------------
# 1. Triton Kernel 实现 (已修复指针计算 Bug)
# -----------------------------------------------------------------------------
@triton.jit
def softmax_tlkernel(
    X,
    Y,
    stride_x,
    stride_y,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # 1. 获取当前处理的行号
    row_idx = tl.program_id(0)
    
    # 2. 计算当前行的指针位置
    x_row_ptr = X + row_idx * stride_x
    y_row_ptr = Y + row_idx * stride_y
    
    # 3. 生成列偏移量 [0, 1, ..., BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    # 4. 生成掩码，防止越界 (处理 N 不是 2 的幂次的情况)
    mask = offsets < N
    
    # 5. 加载数据
    # BUG FIX: 原代码 x_ptr = x_row_ptr + mask 是错误的
    x_ptr = x_row_ptr + offsets
    
    # 加载输入行数据，越界部分填充负无穷大（不影响 Max）
    input_val = tl.load(x_ptr, mask=mask, other=-float('inf')).to(tl.float32)
    
    # 6. Online Softmax 逻辑 (Safe Softmax)
    # 找到当前行的最大值
    max_val = tl.max(input_val, axis=0)
    # 减去最大值，数值稳定性优化
    input_val = input_val - max_val
    
    # 计算分子 (exp)
    numerator = tl.exp(input_val)
    # 计算分母 (sum)
    denominator = tl.sum(numerator, axis=0)
    
    # 7. 计算最终结果
    y = numerator / denominator
    
    # 8. 写回结果
    y_ptrs = y_row_ptr + offsets
    tl.store(y_ptrs, y, mask=mask)

# -----------------------------------------------------------------------------
# 2. Python 包装函数
# -----------------------------------------------------------------------------
def softmax(x):
    M, N = x.shape
    # Block Size 取大于 N 的最小 2 的幂次
    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    
    # 每个 Program 处理一行
    grid = (M, )
    
    # 设置 num_warps 以优化大 Block 的性能
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_tlkernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y

# -----------------------------------------------------------------------------
# 3. 性能基准测试与理论对比分析
# -----------------------------------------------------------------------------
def benchmark_on_h100():
    # H100 SXM5 参数
    GPU_NAME = torch.cuda.get_device_name(0)
    # 如果是 H100 SXM5, 理论带宽约 3350 GB/s (3.35 TB/s)
    # 如果是 H100 PCIe, 理论带宽约 2000 GB/s
    THEORETICAL_BW_GBPS = 2000.0 
    
    print(f"Running on GPU: {GPU_NAME}")
    print(f"Target Theoretical Bandwidth: {THEORETICAL_BW_GBPS} GB/s (H100 SXM5 estimate)")
    
    # 测试数据规模 (模拟 Llama-3-70B 的 Attention Score 矩阵)
    # Batch * Head * Seq_Len * Seq_Len
    # 假设我们测试一行巨大的数据或者多行数据
    # 这里为了稳定测速，设置 M 为较的大数值
    BATCH = 32 * 128  # Batch * Heads
    SEQ_LEN = 8192    # Sequence Length (N)
    
    print(f"\n--- Benchmarking Configuration ---")
    print(f"Shape: [{BATCH}, {SEQ_LEN}]")
    print(f"Data Type: FP16 (2 bytes)")
    
    # 准备数据
    x = torch.randn(BATCH, SEQ_LEN, device='cuda', dtype=torch.float16)
    
    # 1. 正确性验证
    y_triton = softmax(x)
    y_torch = torch.softmax(x.float(), dim=1).half() # Torch softmax 在 fp16 下可能不稳定，转 fp32 算完转回
    
    if torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2):
        print("✅ Correctness Check Passed!")
    else:
        print("❌ Correctness Check Failed!")
        print("Max Diff:", (y_triton - y_torch).abs().max().item())

    # 2. 测量实际运行时间 (ms)
    # triton.testing.do_bench 会自动处理 Warmup 和多次测量取平均
    ms = triton.testing.do_bench(lambda: softmax(x))
    
    # 3. 计算实际吞吐量 (GB/s)
    # Online Softmax 访存量: Read X (2N) + Write Y (2N) = 4N Bytes per row
    # Total Bytes = 4 * M * N
    total_bytes = 4 * BATCH * SEQ_LEN
    actual_bw_gbps = (total_bytes * 1e-9) / (ms * 1e-3)
    
    # 4. 计算理论时间 (ms)
    # Time = Total Traffic / Bandwidth
    theoretical_ms = (total_bytes * 1e-9) / THEORETICAL_BW_GBPS * 1000
    
    # 5. 打印对比报告
    print(f"\n--- Performance Results ---")
    print(f"Actual Runtime       : {ms:.4f} ms")
    print(f"Theoretical Runtime  : {theoretical_ms:.4f} ms (Based on {THEORETICAL_BW_GBPS} GB/s)")
    print(f"Actual Bandwidth     : {actual_bw_gbps:.2f} GB/s")
    print(f"Bandwidth Utilization: {actual_bw_gbps / THEORETICAL_BW_GBPS * 100:.2f}%")
    
    print(f"\n--- Analysis ---")
    if actual_bw_gbps / THEORETICAL_BW_GBPS > 0.75:
        print("🚀 Excellent! The kernel is Memory Bound and highly efficient.")
    else:
        print("⚠️  Room for improvement. Consider tuning num_warps or checking memory coalescing.")

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        benchmark_on_h100()
    else:
        print("CUDA not available, cannot run benchmark.")