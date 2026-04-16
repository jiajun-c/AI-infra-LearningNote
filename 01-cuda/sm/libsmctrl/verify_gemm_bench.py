"""
验证 libsmctrl SM 限制 - GEMM 版本
使用 cuBLAS 进行更稳定的性能测试
"""
import ctypes
import torch
import time

# 加载 libsmctrl
libsmctrl = ctypes.CDLL("/volume/code/jjcheng/libsmctrl/libsmctrl.so")

def set_sm_limit(num_sms):
    """设置 SM 数量限制"""
    if num_sms >= 132:
        libsmctrl.libsmctrl_set_global_mask(0)
    else:
        mask = ~((1 << num_sms) - 1) & 0xFFFFFFFFFFFFFFFF
        libsmctrl.libsmctrl_set_global_mask(mask)

def benchmark_gemm_cublas(sm_count, M, N, K, warmup=10, iterations=50):
    """使用 cuBLAS benchmark GEMM"""
    set_sm_limit(sm_count)

    # 创建矩阵
    A = torch.randn(M, K, dtype=torch.float32).cuda()
    B = torch.randn(K, N, dtype=torch.float32).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    # warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    # 正式测试
    start = time.perf_counter()
    for _ in range(iterations):
        torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed * 1000) / iterations
    tflops = (2 * M * N * K) / (avg_ms / 1000) / 1e12

    return avg_ms, tflops


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 固定 GEMM 形状
    M, N, K = 4096, 4096, 4096
    print(f"GEMM Shape: {M} x {N} x {K}")
    print()

    sm_counts = [132, 96, 64, 48, 32, 24, 16, 8]
    results = []

    print(f"{'SM 数量':>8} | {'耗时 (ms)':>12} | {'TFLOPS':>10} | {'相对性能':>10}")
    print("-" * 50)

    # 先跑 132 SM 作为基准
    print("Running baseline (132 SMs)...")
    set_sm_limit(132)
    baseline_ms, baseline_tflops = benchmark_gemm_cublas(132, M, N, K, warmup=10, iterations=50)
    print(f"     132 | {baseline_ms:>12.3f} | {baseline_tflops:>10.2f} |     100.0% (baseline)")
    print("-" * 50)

    results = [(132, baseline_ms, baseline_tflops)]

    for sm_count in [96, 64, 48, 32, 24, 16, 8]:
        try:
            avg_ms, tflops = benchmark_gemm_cublas(sm_count, M, N, K)
            rel_perf = (tflops / baseline_tflops) * 100
            results.append((sm_count, avg_ms, tflops))
            print(f"{sm_count:>8} | {avg_ms:>12.3f} | {tflops:>10.2f} | {rel_perf:>9.1f}%")
        except Exception as e:
            print(f"{sm_count:>8} | 错误：{e}")

    # 恢复
    set_sm_limit(132)

    print("\n===== 总结 =====")
    print("如果 libsmctrl 生效，减少 SM 数量应该导致性能下降")
    print("对于 compute-bound 的 GEMM，性能下降应该大致与 SM 数量成正比")
