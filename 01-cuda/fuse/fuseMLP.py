import torch
import triton
import triton.language as tl

# ========================================================
# 1. 之前实现的 Triton 融合算子 (Fused MLP)
# ========================================================
@triton.jit
def fused_mlp_gemv_1d_kernel(
    x_ptr, w1_ptr, w2_ptr, y_ptr,
    d_in: tl.constexpr, d_hidden: tl.constexpr, d_out: tl.constexpr,
    stride_w1_in, stride_w1_hid, stride_w2_hid, stride_w2_out,
    BLOCK_IN: tl.constexpr, BLOCK_HIDDEN: tl.constexpr, BLOCK_OUT: tl.constexpr,
):
    # 改为 1D Grid: 仅沿着 Hidden 维度切块
    pid_hidden = tl.program_id(0)

    # 1. 阶段一：算局部 H_inter
    offs_in = tl.arange(0, BLOCK_IN)
    offs_hid = pid_hidden * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    
    acc_h = tl.zeros([BLOCK_HIDDEN], dtype=tl.float32)
    
    # 遍历 D_IN，读取一次 X 和局部 W1
    for i in range(0, d_in, BLOCK_IN):
        curr_offs_in = i + offs_in
        x = tl.load(x_ptr + curr_offs_in, mask=curr_offs_in < d_in, other=0.0)
        
        w1_ptrs = w1_ptr + (curr_offs_in[:, None] * stride_w1_in + offs_hid[None, :] * stride_w1_hid)
        w1 = tl.load(w1_ptrs, mask=(curr_offs_in[:, None] < d_in) & (offs_hid[None, :] < d_hidden), other=0.0)
        
        acc_h += tl.sum(x[:, None] * w1, axis=0)
        
    # 2. 阶段二：原地激活 (SRAM 内)
    h_act = acc_h * tl.sigmoid(acc_h)
    
    # 3. 阶段三：拿着 h_act 遍历 W2 的所有列，算出 Y 的局部结果并原子累加
    offs_out = tl.arange(0, BLOCK_OUT)
    for j in range(0, d_out, BLOCK_OUT):
        curr_offs_out = j + offs_out
        
        # 加载 W2 的一小块 [BLOCK_HIDDEN, BLOCK_OUT]
        w2_ptrs = w2_ptr + (offs_hid[:, None] * stride_w2_hid + curr_offs_out[None, :] * stride_w2_out)
        w2 = tl.load(w2_ptrs, mask=(offs_hid[:, None] < d_hidden) & (curr_offs_out[None, :] < d_out), other=0.0)
        
        # h_act [BLOCK_HIDDEN] @ w2 [BLOCK_HIDDEN, BLOCK_OUT] -> acc_y [BLOCK_OUT]
        acc_y = tl.sum(h_act[:, None] * w2, axis=0)
        
        y_ptrs = y_ptr + curr_offs_out
        mask_y = curr_offs_out < d_out
        
        # 写入 HBM
        tl.atomic_add(y_ptrs, acc_y, mask=mask_y)


def fused_mlp_gemv(x, w1, w2):
    d_in, d_hidden = w1.shape
    d_out = w2.shape[1]
    y = torch.zeros((d_out,), device=x.device, dtype=torch.float32)
    
    BLOCK_IN = 64
    BLOCK_HIDDEN = 64
    BLOCK_OUT = 64
    
    # 启动 1D Grid
    grid = lambda meta: (triton.cdiv(d_hidden, meta['BLOCK_HIDDEN']), )
    
    fused_mlp_gemv_1d_kernel[grid](
        x, w1, w2, y,
        d_in, d_hidden, d_out,
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        BLOCK_IN=BLOCK_IN, BLOCK_HIDDEN=BLOCK_HIDDEN, BLOCK_OUT=BLOCK_OUT,
    )
    return y.to(x.dtype)

# ========================================================
# 2. PyTorch 原生基线 (Unfused MLP)
# ========================================================
def unfused_pytorch_mlp(x, w1, w2):
    # fc1 -> act -> fc2
    h = torch.matmul(x, w1)
    h_act = torch.nn.functional.silu(h)
    y = torch.matmul(h_act, w2)
    return y


# ========================================================
# 3. 性能评测与正确性验证
# ========================================================
def run_benchmark():
    import math # 记得导入 math
    
    D_IN = 4096
    D_HIDDEN = 11008
    D_OUT = 4096
    
    print(f"📦 测试配置: GEMV (Batch=1), D_in={D_IN}, D_hidden={D_HIDDEN}, D_out={D_OUT}")
    
    torch.manual_seed(42)
    
    # 👇 修复 1: 模拟真实模型的权重缩放 (Kaiming / Xavier 思想)
    # 保证每一层输出的方差控制在 1 左右，防止 FP16 精度雪崩
    x = torch.randn(D_IN, dtype=torch.float16, device='cuda')
    w1 = torch.randn(D_IN, D_HIDDEN, dtype=torch.float16, device='cuda') / math.sqrt(D_IN)
    w2 = torch.randn(D_HIDDEN, D_OUT, dtype=torch.float16, device='cuda') / math.sqrt(D_HIDDEN)

    # 1. 验证精度对齐
    out_torch = unfused_pytorch_mlp(x, w1, w2)
    out_triton = fused_mlp_gemv(x, w1, w2)
    
    # 👇 修复 2: 使用深度学习 FP16 算子测试的标准容差
    # FP16 下累加几万次，atol 设为 1e-1 或 5e-2 是业界的 Standard Practice
    try:
        assert torch.allclose(out_torch, out_triton, atol=5e-2, rtol=1e-2)
        print("✅ 精度验证通过！\n")
    except AssertionError:
        # 如果依然报错，打出最大误差看看差距在哪
        max_diff = torch.max(torch.abs(out_torch - out_triton))
        print(f"❌ 精度验证失败！最大绝对误差: {max_diff.item():.4f}")
        return

    # 2. 性能压测 (保持不变)
    quantiles = [0.5, 0.2, 0.8]
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: unfused_pytorch_mlp(x, w1, w2), quantiles=quantiles
    )
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: fused_mlp_gemv(x, w1, w2), quantiles=quantiles
    )

    speedup = ms_torch / ms_triton
    print(f"📊 [PyTorch Unfused] 耗时: {ms_torch * 1000:>6.2f} µs")
    print(f"🚀 [Triton Fused]    耗时: {ms_triton * 1000:>6.2f} µs")
    print(f"{'-'*40}")
    print(f"⚡ 加速比 (Speedup): {speedup:.2f}x")
    
if __name__ == "__main__":
    run_benchmark()