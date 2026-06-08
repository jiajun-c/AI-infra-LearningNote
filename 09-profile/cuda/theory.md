# 理论性能分析

## 1. Flops 和带宽

### 单位换算

PFlops = 1e3 TFlops = 1e3 GFlops = 1e3 MFlops = 1e3 KFlops

一般 GPU 使用 TFlops 衡量算力，使用 GB/s 衡量带宽。

### 理论峰值 GFlops 计算

GPU 的理论峰值浮点性能由硬件规格决定，核心公式为：

```text
理论峰值 GFlops = SM数量 × 每SM的核心数 × GPU频率(GHz) × 每核心每周期操作数
```

其中关键点在于 **FMA (Fused Multiply-Add)** 指令——一次乘加操作算作 **2 次浮点运算**，因此每核心每周期操作数通常为 2。

不同精度的计算公式：

| 精度 | 计算单元 | 公式 |
| ---- | -------- | ---- |
| FP32 | CUDA Core | `SM数 × CUDA核/SM × 频率(GHz) × 2` |
| FP64 | FP64 Unit | `SM数 × FP64核/SM × 频率(GHz) × 2` |
| FP16 (Tensor Core) | Tensor Core | `SM数 × TC/SM × 频率 × 各架构单周期吞吐` |

**以 A100 (GA100) 为例：**

- SM 数量：108
- CUDA Core/SM：64
- 频率：1.41 GHz
- FP32 峰值：108 × 64 × 1.41 × 2 = **19.5 TFLOPS**
- Tensor Core FP16：108 × 4 × 1.41 × 256 = **312 TFLOPS**

**以 H100 (GH100) 为例：**

- SM 数量：132
- CUDA Core/SM：128
- 频率：1.98 GHz
- FP32 峰值：132 × 128 × 1.98 × 2 = **66.9 TFLOPS**
- FP8 (Tensor Core)：**1979 TFLOPS**

### 实际达到的 GFlops 计算

实际运行中，通过测量 kernel 执行时间和浮点运算量来计算：

```text
实际 GFlops = 总浮点运算量(FLOPs) / 执行时间(秒) / 10^9
```

其中：

- **总浮点运算量**：由算子的计算逻辑决定，例如矩阵乘法 `C[M×N] = A[M×K] × B[K×N]` 需要 `2 × M × N × K` 次运算
- **执行时间**：通过 `cudaEvent`、`nsys` profiler 或 `torch.cuda.Event` 测量得到 kernel 耗时

实际 GFlops 与理论峰值的比值即为 **计算利用率**，是衡量 kernel 优化程度的重要指标。

## 2. 计算强度分析

算术强度（Arithmetic Intensity, AI）定义为：

```text
AI = FLOPs / Bytes
```

即每个字节访存对应的浮点运算次数。

**以 online softmax 算子为例：**

- 每个元素需要一次 max、sub、exp、div、add，共约 5~10 FLOPs
- 访存量（FP16）：读写各一次，共 4 Bytes/element
- 算术强度 (AI)：10 / 4 = **2.5 FLOPs/Bytes**

**Roofline 拐点：**

在 H100 上，计算能力为 312 TFLOPS，带宽能力为 2039 GB/s：

```text
计算/访存拐点 = 312 × 10^3 / 2039 ≈ 153 FLOPs/Bytes
```

online softmax 的 AI (2.5) 远小于拐点 (153)，因此是典型的**访存瓶颈**——只考虑访存时间即可：

```text
估计时间 = 2 × N / bandwidth
```

其中 `2 × N` 来自 HBM 上对 N 个元素的读写各一次。

## 3. 实际执行中的额外因素

理论分析之外，实际性能还受以下因素影响：

- **Kernel Launch 开销**：每次 kernel 启动有固定开销（~5-10μs），小 kernel 频繁 launch 时不可忽略
- **Wave 效应**：例如总数据量为 5，但每次只能处理 4 个元素，第二波只处理 1 个——最后一波计算资源利用率不足

> 进一步的分析见 [Warp Stall 分析](./stall.md)，通过硬件计数器判断 kernel 实际是 compute-bound 还是 memory-bound。
