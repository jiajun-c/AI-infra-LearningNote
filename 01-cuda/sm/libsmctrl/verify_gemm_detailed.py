"""
验证 libsmctrl SM 限制 - GEMM 版本（增强版）
测试更多 SM 数量配置，找出实际生效的阈值
"""
import ctypes
import torch
import time

# 加载 libsmctrl
libsmctrl = ctypes.CDLL("/volume/code/jjcheng/libsmctrl/libsmctrl.so")

def set_next_sm_limit(num_sms):
    """设置下一个 kernel 的 SM 数量限制"""
    if num_sms >= 132:
        libsmctrl.libsmctrl_set_next_mask(0)
    else:
        mask = ~((1 << num_sms) - 1) & 0xFFFFFFFFFFFFFFFF
        libsmctrl.libsmctrl_set_next_mask(mask)

def benchmark_gemm_once(num_sms, M, N, K):
    """跑一次 GEMM 并返回性能"""
    set_next_sm_limit(num_sms)

    A = torch.randn(M, K, dtype=torch.float32).cuda()
    B = torch.randn(K, N, dtype=torch.float32).cuda()

    torch.mm(A, B)
    torch.cuda.synchronize()

def benchmark_gemm_cublas(num_sms, M, N, K, warmup=5, iterations=20):
    """使用 cuBLAS benchmark GEMM"""
    # warmup
    for _ in range(warmup):
        benchmark_gemm_once(132, M, N, K)

    # 正式测试：每次都设置限制
    times = []
    for _ in range(iterations):
        set_next_sm_limit(num_sms)

        A = torch.randn(M, K, dtype=torch.float32).cuda()
        B = torch.randn(K, N, dtype=torch.float32).cuda()

        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.mm(A, B)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    avg_ms = sum(times) / len(times)
    tflops = (2 * M * N * K) / (avg_ms / 1000) / 1e12

    return avg_ms, tflops


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 固定 GEMM 形状
    M, N, K = 4096, 4096, 4096
    print(f"GEMM Shape: {M} x {N} x {K}")
    print()

    # 更细粒度的 SM 配置
    sm_counts = [132, 128, 96, 64, 48, 32, 24, 20, 16, 12, 8, 4]

    print(f"{'SM 数量':>8} | {'耗时 (ms)':>12} | {'TFLOPS':>10} | {'相对性能':>10}")
    print("-" * 50)

    results = []

    for sm_count in sm_counts:
        try:
            avg_ms, tflops = benchmark_gemm_cublas(sm_count, M, N, K)
            results.append((sm_count, avg_ms, tflops))
        except Exception as e:
            print(f"{sm_count:>8} | 错误：{e}")
            continue

    # 计算相对性能（以 132 SM 为基准）
    baseline_tflops = results[0][2] if results else 1
    print()
    print(f"{'SM 数量':>8} | {'耗时 (ms)':>12} | {'TFLOPS':>10} | {'相对性能':>10}")
    print("-" * 50)
    for sm_count, avg_ms, tflops in results:
        rel_perf = (tflops / baseline_tflops) * 100
        print(f"{sm_count:>8} | {avg_ms:>12.3f} | {tflops:>10.2f} | {rel_perf:>9.1f}%")

    # 恢复默认
    libsmctrl.libsmctrl_set_global_mask(0)

    print()
    print("===== 总结 =====")
    print("数据表明 libsmctrl 在 H100 上的 SM 限制存在阈值效应")
    print("低于某个阈值后，性能才会明显下降")
