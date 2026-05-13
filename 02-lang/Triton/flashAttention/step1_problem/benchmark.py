"""
Step 1: 验证标准 Attention 的内存瓶颈

运行方式: python benchmark.py
"""

import torch
import torch.nn.functional as F
import time


def standard_attention(q, k, v):
    """标准 Attention 实现"""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / (q.size(-1) ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


def analyze_memory(seq_len, head_dim=64, dtype=torch.float16):
    """分析内存使用情况"""
    element_size = 2 if dtype == torch.float16 else 4

    # 输入大小
    input_size = seq_len * head_dim * element_size

    # 中间矩阵大小
    scores_size = seq_len * seq_len * element_size
    attn_weights_size = seq_len * seq_len * element_size

    print(f"\n{'='*50}")
    print(f"序列长度: {seq_len}, head_dim: {head_dim}")
    print(f"{'='*50}")
    print(f"Q, K, V 每个大小: {input_size / 1024:.2f} KB")
    print(f"scores 矩阵大小: {scores_size / 1024 / 1024:.2f} MB")
    print(f"attn_weights 矩阵大小: {attn_weights_size / 1024 / 1024:.2f} MB")
    print(f"中间结果总计: {(scores_size + attn_weights_size) / 1024 / 1024:.2f} MB")
    print(f"内存复杂度: O(N²) = {seq_len**2} 元素")


def benchmark_attention(seq_len, head_dim=64, num_heads=12, batch_size=1, dtype=torch.float16, num_iterations=100):
    """性能基准测试"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print("\n警告: 未检测到 GPU, 使用 CPU 运行")
        return None

    # 创建输入
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    # 预热
    for _ in range(10):
        _ = standard_attention(q, k, v)
    torch.cuda.synchronize()

    # 计时
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = standard_attention(q, k, v)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations * 1000  # ms

    print(f"\n序列长度 {seq_len}:")
    print(f"  平均执行时间: {avg_time:.3f} ms")
    print(f"  吞吐量: {seq_len / avg_time * 1000:.0f} tokens/sec")

    return avg_time


def main():
    print("=" * 60)
    print("Step 1: 标准 Attention 内存瓶颈分析")
    print("=" * 60)

    # 内存分析
    print("\n📊 内存使用分析:")
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        analyze_memory(seq_len)

    # 性能基准测试
    print("\n⏱️ 性能基准测试:")
    for seq_len in [512, 1024, 2048, 4096]:
        benchmark_attention(seq_len)

    print("\n" + "=" * 60)
    print("结论:")
    print("  - 内存使用随 N² 增长")
    print("  - 长序列 (N=4096+) 中间结果超过 32MB")
    print("  - 需要频繁 HBM 读写，成为性能瓶颈")
    print("=" * 60)


if __name__ == "__main__":
    main()