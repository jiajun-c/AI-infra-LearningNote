import torch
import triton
import triton.language as tl

# ========================================================
# 1. 你提供的 Naive 版 Triton GEMV
# ========================================================
@triton.jit
def _gemv_naive(
    x_ptr, A_ptr, y_ptr,
    N, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 按照 N 维度切分，每个 Block 负责一行
    n = tl.program_id(0)
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask = offs_k < K
    
    # 假设 A 是 Row-Major (行主序)，偏移量计算为 n * K + offs_k
    a_ptrs = A_ptr + n * K + offs_k
    
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offs_k, mask=mask, other=0.0)
    
    dot = tl.sum(a_vals * x_vals, axis=0)
    tl.store(y_ptr + n, dot)

# Python 包装函数
def triton_gemv_naive(x, A):
    assert x.ndim == 1 and A.ndim == 2
    K = x.shape[0]
    N = A.shape[0] # 注意：根据你的指针逻辑 a_ptrs = A_ptr + n * K，A 的形状必须是 [N, K]
    
    y = torch.zeros((N,), device=x.device, dtype=torch.float32)
    
    # 获取大于等于 K 的最小 2 的幂次方
    BLOCK_SIZE_K = triton.next_power_of_2(K)
    
    # 启动 1D Grid，大小为 N
    grid = (N,)
    
    _gemv_naive[grid](
        x, A, y,
        N, K,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return y

# ========================================================
# 2. PyTorch 原生基线 (cuBLAS)
# ========================================================
def torch_gemv(x, A):
    # A 是 [N, K], x 是 [K], 结果是 [N]
    # torch.mv 专门用于矩阵向量乘法
    return torch.mv(A, x)

# ========================================================
# 3. 性能评测与正确性验证
# ========================================================
def run_benchmark():
    # ⚠️ 为什么不用 LLaMA 的 K=11008？
    # 因为 Naive 内核没有 for 循环，试图一次性加载 K！
    # 如果 K 太大，BLOCK_SIZE_K 会变成 16384，直接撑爆 GPU 寄存器导致编译失败或极慢。
    N = 4096
    K = 4096 
    
    print(f"📦 测试配置: Naive GEMV, N (输出)={N}, K (输入)={K}")
    
    torch.manual_seed(42)
    # 使用 FP32 测试，因为你的 kernel 累加没有做精度转换
    x = torch.randn(K, dtype=torch.float32, device='cuda')
    A = torch.randn(N, K, dtype=torch.float32, device='cuda')

    # 1. 验证精度对齐
    out_torch = torch_gemv(x, A)
    out_triton = triton_gemv_naive(x, A)
    
    assert torch.allclose(out_torch, out_triton, atol=1e-2, rtol=1e-2), "❌ 精度验证失败！"
    print("✅ 精度验证通过！\n")

    # 2. 性能压测
    quantiles = [0.5, 0.2, 0.8]
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: torch_gemv(x, A), quantiles=quantiles
    )
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: triton_gemv_naive(x, A), quantiles=quantiles
    )

    speedup = ms_torch / ms_triton
    print(f"📊 [PyTorch (cuBLAS)] 耗时: {ms_torch * 1000:>6.2f} µs")
    print(f"🐌 [Triton Naive]     耗时: {ms_triton * 1000:>6.2f} µs")
    print(f"{'-'*40}")
    print(f"⚡ 加速比 (Speedup): {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()