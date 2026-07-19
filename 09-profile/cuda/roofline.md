# Roofline 分析

Roofline Model 由 UC Berkeley 的 Samuel Williams 等人在 2008 年提出，是性能分析领域最核心的可视化工具之一。它用一张图回答两个问题：

1. **这个 kernel 能跑多快？**（硬件上限）
2. **为什么实际跑不到这么快？**（瓶颈在哪）

## 1. 模型构成

Roofline 图以**算术强度（Arithmetic Intensity, AI）**为横轴，**吞吐量（FLOPs/s）**为纵轴：

```text
                ↑ 吞吐量 (FLOPs/s)
                │
    峰值算力 ───├────────────────────────────┐
                │          Compute Bound      │
                │         (计算瓶颈区)         │
                │                              │
                │         ╱                    │
                │       ╱                      │
                │     ╱  Memory Bound          │
                │   ╱   (访存瓶颈区)            │
                │ ╱                            │
                └──────────────────────────────→
                        算术强度 (FLOPs/Byte)
                             ↑
                        拐点 (Ridge Point)
```

两条边界线构成了"roof"（屋顶），kernel 的位置只能在 roof 之下：

| 边界 | 公式 | 瓶颈 |
| ---- | ---- | ---- |
| **带宽线**（斜线） | `P = AI × Bandwidth` | Memory-bound：跑不满算力，被带宽卡住 |
| **算力线**（水平线） | `P = Peak FLOPs` | Compute-bound：带宽够用，但算力已到上限 |

两条线的交点即**拐点（Ridge Point）**：

```text
Ridge Point = Peak FLOPs / Peak Bandwidth
```

- AI < 拐点 → 在带宽线下方 → **Memory-bound**
- AI ≥ 拐点 → 在算力线下方 → **Compute-bound**

## 2. 多精度 Roofline

同一 GPU 在不同精度下有完全不同的 roofline。以 **H100 (GH100)** 为例：

| 精度 | 峰值算力 | 峰值带宽 | 拐点 |
| ---- | -------- | -------- | ---- |
| FP64 Tensor Core | 66.9 TFLOPS | 2039 GB/s | ~33 FLOPs/Byte |
| FP32 CUDA Core | 66.9 TFLOPS | 2039 GB/s | ~33 FLOPs/Byte |
| FP16 Tensor Core | 989 TFLOPS | 2039 GB/s | ~485 FLOPs/Byte |
| FP8 Tensor Core | 1979 TFLOPS | 2039 GB/s | ~970 FLOPs/Byte |
| INT8 Tensor Core | 1979 TOPS | 2039 GB/s | ~970 OPs/Byte |

**关键洞察**：精度越低，拐点越高。FP8 比 FP32 的拐点高约 30 倍，意味着低精度更容易陷入 memory-bound。这也解释了为什么 FP8/FP16 推理时需要大量 tiling 和 fusion 来减少访存。

### H100 Roofline 示意图

```text
FP8 算力  ──────────────────────────────────────── 1979 TFLOPS
FP16 算力 ──────────────────────────────── 989 TFLOPS
                                            ↗ FP8 斜率 = 2039 GB/s
                                       ↗ FP16 斜率 = 2039 GB/s
                                  ↗ 所有精度共用同一带宽线
                             ↗
                        ↗
FP32 算力 ───────── 66.9 TFLOPS
              ↗
            拐点 FP32≈33   拐点 FP16≈485   拐点 FP8≈970
```

> 所有精度共享同一条物理带宽线（2039 GB/s），但不同精度的算力上限不同，导致拐点位置不同。

## 3. 使用 Roofline 分析 Kernel

### 步骤

1. **计算 kernel 的算术强度 AI**
   - 统计 FLOPs：矩阵乘 `C[M×N] = A[M×K] × B[K×N]` → `2 × M × N × K` FLOPs
   - 统计访存字节数：HBM 读写总量
   - `AI = FLOPs / Bytes`

2. **在 roofline 图上标出 kernel 的 (AI, 实测吞吐) 点**

3. **根据位置判断优化方向**：
   - 在带宽线上 → Memory-bound → 减少 HBM 访存
   - 在算力线附近 → Compute-bound → 需要算法层面提效
   - 远低于两条线 → **实现问题**（cache miss、bank conflict、occupancy 低）

## 4. 使用 Roofline 进行CTA层级的分析

可以看到上面的分析往往局限于整个算子的层面，假设需要进行更加细致的分析

例如对于一个使用cuda core和share memory的算子而言，其每个时钟周期完成的访存为 32x4B = 128B，而由于一个CTA内一次可以有四个warp在执行，所以可以执行的FMA的操作数量为4x32=128FMA，一个FMA等于两个普通操作。

所以rigde的点为AI = 128FMA/cycle/128B/cycle = 1MFA/B = 2

只有当计算强度>2的时候才能喂满cuda core



### 典型算子分类

| 算子 | 近似 AI | 在 H100 上的瓶颈 |
| ---- | ------- | ----------------- |
| Element-wise (add, relu) | ~0.25 | **Memory-bound**（极低 AI） |
| LayerNorm/RMSNorm | ~0.3~0.5 | **Memory-bound** |
| Softmax (naive) | ~1~3 | **Memory-bound** |
| Online Softmax | ~2.5 | **Memory-bound** |
| FlashAttention (fwd) | ~50~100 | 接近拐点，需具体分析 |
| MatMul (大尺寸) | ~N/4 (N=K维) | 当 N 足够大时 **Compute-bound** |
| Conv 3×3 | ~10~50 | 取决于实现，可能双瓶颈 |

## 4. 优化策略

### Memory-bound kernel（AI < 拐点）

目标是**提高算术强度**，让每个字节的访存做更多计算：

| 技术 | 原理 | 效果 |
| ---- | ---- | ---- |
| **Kernel Fusion** | 将多个 element-wise kernel 合并，避免中间结果写回 HBM | 减少访存次数 |
| **Tiling** | 将数据切成小块，用 shared memory 缓存复用 | HBM 访问 → SMEM 访问 |
| **Vectorized Load** | `float4`/`uint4` 一次读取 128bit | 充分利用总线带宽 |
| **Coalesced Access** | 让同一 warp 的线程访问连续地址 | 减少 memory transaction 数 |
| **FP8/INT8 量化** | 减少每个元素的字节数 | 降低总 Bytes |

### Compute-bound kernel（AI ≥ 拐点）

目标是**提高算力利用率**：

| 技术 | 原理 | 效果 |
| ---- | ---- | ---- |
| **Tensor Core** | 专用矩阵乘累加单元 | FP16 算力 ↑ 10-30× |
| **Warp Tiling** | 每个 warp 处理更大的 tile | 提高 instruction-level parallelism |
| **Software Pipelining** | 异步拷贝 + 计算重叠（Hopper TMA） | 隐藏访存延迟 |
| **双缓冲** | 一个 buffer 计算时另一个加载数据 | 隐藏访存延迟 |
| **提高 Occupancy** | 增加活跃 warp 数 | 隐藏依赖延迟 |

## 5. Roofline 的局限性

1. **假设带宽可以完全打满**：实际利用率受访问模式影响（随机访问 vs 连续访问）
2. **不区分内存层次**：L2 cache 和 HBM 的带宽差异巨大，但 roofline 只用 HBM 带宽画线。访存密集 kernel 如果 L2 命中率高，实际性能可以突破"带宽线"
3. **不反映延迟敏感度**：roofline 只看吞吐，低 occupancy 导致的延迟暴露无法体现
4. **Wave 效应和 launch overhead** 不完全反映在 roofline 上

## 6. 与其他分析方法的配合

```text
        理论分析                  硬件验证
        ────────                  ────────

    [Roofline 分析]           [Warp Stall 分析]
    判断 kernel 应该           从 profiler 看实际
    是 compute 还是              Short/Long Stall
    memory bound               占比是否符合预期

            ↘                     ↙
            [一致？] ──否──→ 重新检查 AI 计算或实现问题
              │
             是
              ↓
         确定优化方向
    (fusion/tiling vs TensorCore/occupancy)
```

- [理论性能分析](./theory.md)：GFlops 计算方法和 AI 公式
- [Warp Stall 分析](./stall.md)：通过硬件计数器验证 roofline 的理论判断

## 参考

- Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM.
- NVIDIA Nsight Compute Documentation: *Roofline Analysis*
