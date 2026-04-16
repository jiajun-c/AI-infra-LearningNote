"""
验证 libsmctrl 是否真的限制了 SM 使用量
通过比较不同 SM 限制下的性能来间接验证
"""
import ctypes
import torch
import time

# 加载 libsmctrl
libsmctrl = ctypes.CDLL("libsmctrl.so")

def set_global_sm_count(num_sms):
    """设置全局默认 SM 数量"""
    mask = ~((1 << num_sms) - 1) & 0xFFFFFFFFFFFFFFFF
    libsmctrl.libsmctrl_set_global_mask(mask)

def benchmark_kernel(sm_count, iterations=100):
    """benchmark 并返回平均每迭代时间"""
    set_global_sm_count(sm_count)

    # 更大数据量，让 kernel 运行更久
    size = 500_000_000
    x = torch.randn(size, dtype=torch.float32).cuda()
    y = torch.randn(size, dtype=torch.float32).cuda()

    # warmup
    for _ in range(10):
        z = torch.sin(x) + torch.cos(y)
        z = z * 2.0
    torch.cuda.synchronize()

    # 正式测试
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        z = torch.sin(x) + torch.cos(y)
        z = z * 2.0

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed * 1000) / iterations
    throughput_gb_s = (size * 4 * 3) / (avg_ms / 1000) / 1e9  # 3 次内存访问 (读 x, 读 y, 写 z)

    return avg_ms, throughput_gb_s


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 计算能力：{torch.cuda.get_device_capability()}")
    print()

    sm_counts = [66, 48, 32, 24, 16, 8, 4]
    results = []

    print(f"{'SM 数量':>8} | {'耗时 (ms)':>10} | {'吞吐 (GB/s)':>12} | {'相对性能':>10}")
    print("-" * 50)

    for sm_count in sm_counts:
        try:
            avg_ms, throughput = benchmark_kernel(sm_count)
            results.append((sm_count, avg_ms, throughput))
            rel_perf = (throughput / results[0][2]) * 100
            print(f"{sm_count:>8} | {avg_ms:>10.2f} | {throughput:>12.1f} | {rel_perf:>9.1f}%")
        except Exception as e:
            print(f"{sm_count:>8} | 错误：{e}")

    # 恢复
    set_global_sm_count(66)

    print("\n如果 SM 限制生效，性能应该随 SM 数量减少而下降")
    print("对于 memory-bound kernel，可能在小范围内性能不变")
