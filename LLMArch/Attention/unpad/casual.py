import torch
import torch.nn.functional as F
import time
import math

def benchmark_attention(seq_len, batch_size=4, num_heads=16, head_dim=128, dtype=torch.float16):
    device = torch.device("cuda")
    
    # 1. 准备数据
    # Shape: [Batch, Heads, SeqLen, Dim]
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # 2. 准备 Explicit Mask (用于模拟笨办法)
    # Shape: [SeqLen, SeqLen] -> 广播到 Batch
    # 这是一个巨大的下三角矩阵，右上角是 -inf
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)

    # --- 定义两个对比函数 ---

    # Case A: 智能模式 (is_causal=True)
    # 底层 Kernel 知道是对角线，直接跳过计算
    def run_causal():
        return F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=True 
        )

    # Case B: 笨重模式 (Explicit Mask)
    # 底层必须加载 mask 矩阵，且通常无法有效跳过计算
    def run_explicit_mask():
        return F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            is_causal=False
        )

    # --- 开始 Profile (使用 CUDA Event 计时) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 预热 (Warmup)
    for _ in range(10): run_causal()
    for _ in range(10): run_explicit_mask()
    torch.cuda.synchronize()

    iters = 100

    # 1. 测试 is_causal=True
    start_event.record()
    for _ in range(iters):
        run_causal()
    end_event.record()
    torch.cuda.synchronize()
    time_smart = start_event.elapsed_time(end_event) / iters

    # 2. 测试 Explicit Mask
    start_event.record()
    for _ in range(iters):
        run_explicit_mask()
    end_event.record()
    torch.cuda.synchronize()
    time_mask = start_event.elapsed_time(end_event) / iters

    print(f"| {seq_len:<7} | {time_smart:.3f} ms | {time_mask:.3f} ms | {time_mask / time_smart:.2f}x      |")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: 需要 GPU 才能运行 FlashAttention Profile。")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("-" * 55)
        print(f"| SeqLen  | Causal(Fast) | Masked(Slow) | Speedup |")
        print("-" * 55)
        
        # 测试不同的序列长度，观察差距如何拉大
        for seq_len in [1024, 2048, 4096, 8192]:
            try:
                benchmark_attention(seq_len)
            except torch.cuda.OutOfMemoryError:
                print(f"| {seq_len:<7} | OOM          | OOM          | -       |")