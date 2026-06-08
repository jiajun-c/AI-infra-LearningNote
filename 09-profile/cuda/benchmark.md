# GFlops Benchmark 实战

## 核心思路

```text
实际 GFlops = kernel 的总 FLOPs / kernel 执行时间(秒) / 10^9
```

两个关键量：
1. **FLOPs**——你的 kernel 做了多少次浮点运算（手动计算，必须精确）
2. **执行时间**——用 GPU timer 精确测量（不能用 CPU 时间）

## 1. CUDA C++ 方式（最精确）

### 最小完整例子：测量 GEMM 的 GFlops

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define WARMUP_ITERS 10
#define BENCH_ITERS  100

int main() {
    const int M = 4096, N = 4096, K = 4096;
    float *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));

    // 初始化数据...
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // === warmup：让 GPU 频率稳定 ===
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
    cudaDeviceSynchronize();

    // === 正式计时 ===
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float time_per_kernel = ms / BENCH_ITERS;  // 毫秒

    // === 计算 GFlops ===
    // GEMM FLOPs = 2 * M * N * K
    double flops = 2.0 * M * N * K;
    double gflops = flops / (time_per_kernel / 1000.0) / 1e9;

    printf("GEMM %dx%dx%d: %.2f ms, %.2f GFLOPS\n",
           M, N, K, time_per_kernel, gflops);

    // 与理论峰值对比
    double peak_fp32 = 66.9e3;  // H100 FP32 TFLOPS → GFLOPS
    printf("利用率: %.1f%%\n", 100.0 * gflops / peak_fp32);

    cublasDestroy(handle);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
```

### 手写 Kernel 的 Benchmark 模式

```cpp
// 1. 先跑几轮 warmup（不计时）
for (int i = 0; i < 10; i++) {
    myKernel<<<grid, block>>>(d_in, d_out, N);
}
cudaDeviceSynchronize();

// 2. 用 cudaEvent 精确计时
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
for (int i = 0; i < 100; i++) {
    myKernel<<<grid, block>>>(d_in, d_out, N);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);

// 3. 计算 GFlops
// 你的 kernel 每次做多少 FLOPs？
double flops_per_call = /* 手算 */;
double gflops = flops_per_call / (ms / 100.0) / 1e9;
```

## 2. PyTorch 方式（快速验证）

```python
import torch

def benchmark_gemm(M, N, K, warmup=10, iters=100):
    """测量 PyTorch GEMM 的实际 GFlops"""
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    # === warmup ===
    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()

    # === 计时 ===
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        C = A @ B
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)  # iters 次的总时间
    time_per_call_ms = elapsed_ms / iters

    # GEMM FLOPs = 2 * M * N * K
    flops = 2.0 * M * N * K
    gflops = flops / (time_per_call_ms / 1e3) / 1e9

    print(f"GEMM {M}x{N}x{K}: "
          f"{time_per_call_ms:.3f} ms/kernel, "
          f"{gflops:.1f} GFLOPS")
    return gflops

# 验证不同尺寸的利用率
for M in [1024, 2048, 4096, 8192]:
    benchmark_gemm(M, M, M)
```

### Element-wise Kernel 示例

```python
def benchmark_elementwise(N, warmup=10, iters=100):
    """测量 element-wise 的 GFlops —— 预期很低（memory-bound）"""
    x = torch.randn(N, device='cuda', dtype=torch.float32)

    # warmup
    for _ in range(warmup):
        y = x * 2 + 1
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = x * 2 + 1  # 1次乘法 + 1次加法 = 2 FLOPs
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    total_flops = 2.0 * N * iters      # 每次 2 FLOPs
    gflops = total_flops / (elapsed_ms / 1e3) / 1e9

    # 也计算一下带宽，因为这是个 memory-bound kernel
    bytes_total = N * 4 * 2 * iters     # 读 N*4B + 写 N*4B
    bandwidth_gb_s = bytes_total / (elapsed_ms / 1e3) / 1e9

    print(f"elemwise N={N}: {elapsed_ms:.3f} ms, "
          f"{gflops:.1f} GFLOPS, {bandwidth_gb_s:.1f} GB/s")
```

## 3. Benchmark 的关键注意事项

### Warmup 是必须的

GPU 频率不是固定的，冷启动时频率较低，几次 kernel 调用后才稳定到 boost clock：

```
第 1 次调用:  GPU clock ~ 800 MHz   ← 不准
第 2~5 次:    GPU clock 上升中      ← 不准
第 10 次以后:  GPU clock ~ 1980 MHz  ← 稳定，可以用
```

**至少 warmup 5-10 次再开始计时。**

### GPU 是异步的，必须 synchronize

```python
# ❌ 错误：time.time() 是 CPU 时间，GPU 可能还没执行完
import time
t0 = time.time()
kernel(A, B)
t1 = time.time()  # GPU 可能还在跑！

# ✅ 正确：用 CUDA event 或 synchronize
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
kernel(A, B)
end.record()
torch.cuda.synchronize()  # 等 GPU 跑完
elapsed = start.elapsed_time(end)
```

### FLOPs 计数必须精确

| 操作 | FLOPs |
|------|-------|
| `a + b` | 1 |
| `a * b` | 1 |
| `a * b + c` (FMA) | **2**（一次乘加算两次浮点运算） |
| `C[M×N] += A[M×K] × B[K×N]` | **2 × M × N × K**（M×N×K 次乘 + M×N×K 次加） |
| `exp(x)` | ~8-16（超越函数，非单次操作） |
| `sqrt(x)` | ~4-8 |

### 多次迭代取平均

单次 kernel 执行时间只有几微秒到几毫秒，误差大。做法：

```cpp
// 让循环跑 100 次，总时间除以 100
// 而不是用 100 次的平均值（cudaEvent 有开销）
for (int i = 0; i < 100; i++) {
    kernel<<<...>>>(...);
}
// 总时间 / 100 = 单次平均
```

### 报告什么指标

| 指标 | 公式 | 含义 |
| ---- | ---- | ---- |
| **GFlops** | FLOPs / time / 1e9 | 实际算力吞吐 |
| **利用率** | GFlops / Peak_GFlops × 100% | 达到峰值算力的百分之几 |
| **带宽** | Bytes / time / 1e9 (GB/s) | 实际访存带宽 |
| **AI** | FLOPs / Bytes | 算术强度，决定瓶颈 |

一个 kernel 应该同时报告 GFlops 和带宽，才能判断瓶颈在哪。

## 4. 完整 Benchmark 脚本模板

```python
import torch

def bench(name, fn, flops_per_call, bytes_per_call,
          warmup=10, iters=100):
    """通用 benchmark 模板"""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    gflops = flops_per_call / (elapsed_ms / 1e3) / 1e9
    bw = bytes_per_call / (elapsed_ms / 1e3) / 1e9 if bytes_per_call else 0
    ai = flops_per_call / bytes_per_call if bytes_per_call else float('inf')

    print(f"{name:20s} | {elapsed_ms:8.3f}ms | "
          f"{gflops:8.1f} GFLOPS | {bw:7.1f} GB/s | AI={ai:.1f}")

# --- 使用示例 ---
N = 2**20  # 1M elements
x = torch.randn(N, device='cuda', dtype=torch.float32)
y = torch.randn(N, device='cuda', dtype=torch.float32)
out = torch.empty(N, device='cuda', dtype=torch.float32)

# Element-wise add: 1 FLOP, 12 bytes (2读+1写)
bench("elem_add",
      lambda: x.add(y, out=out),
      flops_per_call=N,
      bytes_per_call=N * 3 * 4)  # 3 × 4B

# Element-wise fused: 3 FLOPs (mul + add + relu), 12 bytes
bench("fused_mul_add_relu",
      lambda: torch.nn.functional.relu(x * 2 + y, inplace=False),
      flops_per_call=N * 3,
      bytes_per_call=N * 4 * 3)
```

运行后输出示例（H100）：

```text
name                 |     ms    |    GFLOPS |    GB/s | AI
elem_add             |    0.021  |     50.0  |  600.0  | 0.08
fused_mul_add_relu   |    0.022  |    142.9  |  571.4  | 0.25
```

两个 kernel 的 AI 都远低于 H100 拐点（~33），所以 GFLOPS 很低但带宽接近峰值——典型的 memory-bound。

## 5. 常见陷阱

| 陷阱 | 现象 | 解决 |
|------|------|------|
| **编译器优化掉了你的 kernel** | GFLOPS 高得离谱 | 让输出 `volatile` 或打印结果 |
| **数据在 L2 cache 里** | 第一次很慢，后面很快 | 每次迭代用不同数据，或 memset 清 cache |
| **Grid/Block 太小** | GFLOPS 远低于预期 | 确保 `grid_size ≫ SM 数量`，让 GPU 满载 |
| **I-cache miss** | 前几次慢 | 已经 warmup 了，应该 OK |
| **cudaEvent 开销在循环里** | 测量值偏低（时间偏长） | 不要把 `record` 放在循环内；循环外包一次 start/stop |
| **CPU timer 代替 GPU timer** | 时间不准 | 必须用 `cudaEvent` 或 `torch.cuda.Event`，不能用 `time.time()` |

## 与其他文档的关系

- [理论分析](./theory.md)：理论峰值 GFlops 的计算公式
- [Roofline 分析](./roofline.md)：根据测出的 AI 和 GFlops 画到 roofline 图上
- [Warp Stall 分析](./stall.md)：如果 GFLOPS 远低于理论值，用 stall 分析定位原因
