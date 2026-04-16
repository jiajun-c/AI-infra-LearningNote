"""
验证 libsmctrl 是否真的限制了 SM 使用量
通过比较不同 SM 限制下的 GEMM 性能来间接验证
"""
import ctypes
import torch
import time

# 加载 libsmctrl
libsmctrl = ctypes.CDLL("/volume/code/jjcheng/libsmctrl/libsmctrl.so")

def set_global_sm_count(num_sms):
    """设置全局默认 SM 数量"""
    mask = ~((1 << num_sms) - 1) & 0xFFFFFFFFFFFFFFFF
    libsmctrl.libsmctrl_set_global_mask(mask)

def benchmark_gemm(sm_count, M, N, K, iterations=20):
    """benchmark GEMM 并返回平均每迭代时间"""
    set_global_sm_count(sm_count)

    # 创建矩阵
    A = torch.randn(M, K, dtype=torch.float32).cuda()
    B = torch.randn(K, N, dtype=torch.float32).cuda()

    # warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # 正式测试
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed * 1000) / iterations

    # GEMM 浮点运算次数：2 * M * N * K
    flop = 2 * M * N * K
    tflops = flop / (elapsed / iterations) / 1e12

    return avg_ms, tflops


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 计算能力：{torch.cuda.get_device_capability()}")
    print()

    # GEMM 形状：可以选择不同的尺寸
    test_shapes = [
        # (M, N, K)
        (512, 512, 512),      # 小矩阵
        (1024, 1024, 1024),   # 中等矩阵
        (2048, 2048, 2048),   # 大矩阵
        (4096, 4096, 4096),   # 更大矩阵
    ]

    sm_counts = [132, 96, 64, 48, 32, 24, 16, 8]

    for M, N, K in test_shapes:
        print(f"===== GEMM Shape: {M} x {N} x {K} =====")
        print(f"{'SM 数量':>8} | {'耗时 (ms)':>10} | {'TFLOPS':>10} | {'相对性能':>10}")
        print("-" * 45)

        results = []
        for sm_count in sm_counts:
            try:
                avg_ms, tflops = benchmark_gemm(sm_count, M, N, K, iterations=20)
                results.append((sm_count, avg_ms, tflops))
                rel_perf = (tflops / results[0][1]) * 100 if results[0][1] > 0 else 0
                print(f"{sm_count:>8} | {avg_ms:>10.3f} | {tflops:>10.2f} | {rel_perf:>9.1f}%")
            except Exception as e:
                print(f"{sm_count:>8} | 错误：{e}")

        print()

    # 恢复
    set_global_sm_count(132)

    print("\n如果 SM 限制生效，性能应该随 SM 数量减少而下降")
    print("对于 compute-bound 的 GEMM，性能下降应该更明显")
