# SM 微架构

SM（Streaming Multiprocessor）是 GPU 执行计算的基本单元。理解其内部结构是性能优化的基础。

## 1. SM 内部结构

SM 内部进一步分为 **4 个 Subpartition**（子分区），每个 subpartition 是一个独立的指令发射单元：

```text
SM (Streaming Multiprocessor)
│
├── Subpartition 0
│   ├── Warp Scheduler × 1   ── 每周期从就绪 warp 中选一条指令
│   ├── Dispatch Unit  × 1   ── 将指令分发到执行单元
│   ├── CUDA Core (FP32 × 16) ── 单精度浮点/整数运算
│   ├── CUDA Core (INT32 × 16)── 整数运算（可并行执行）
│   ├── Tensor Core × 1      ── 矩阵乘加加速器
│   ├── SFU × 4              ── sin/cos/log/exp 等特殊函数
│   ├── LD/ST Unit × 8       ── 全局/局部内存访问
│   └── Register File        ── 65536 / 4 = 16384 个 32-bit 寄存器
│
├── Subpartition 1 ─┐
├── Subpartition 2  ├── 同 Subpartition 0
├── Subpartition 3 ─┘
│
└── SM 级共享资源
    ├── L1 Cache / Shared Memory: 256 KB (可配置比例)
    ├── L0 Instruction Cache
    └── Constant Cache
```

**关键：** 每周期每个 subpartition 只能发射 1 条 warp 指令。4 个 subpartition 意味着一个 SM **每周期最多发射 4 条 warp 指令**。这就是为什么 SM 能同时管理 64 个 warp —— 通过 4 个 subpartition 各自轮流调度。

## 2. 关键参数演进

| 参数 | Ampere (A100) | Hopper (H100) | Blackwell (B200) |
|------|:-----------:|:-----------:|:--------------:|
| Compute Capability | 8.0 | 9.0 | 10.0 |
| SM 数量 | 108 | 132 | 160 |
| 最大线程/SM | 2048 | 2048 | 2048 |
| 最大 Warp/SM | 64 | 64 | 64 |
| 最大 Block/SM | 32 | 32 | 32 |
| 寄存器总量/SM | 65536 | 65536 | 65536 |
| L1/SMEM 总量 | 192 KB | 256 KB | 256 KB |
| 最大 SMEM/Block | 164 KB | 228 KB | 228 KB |
| L2 Cache | 40 MB | 50 MB | — |

## 3. 缓存层次

### 每 SM 内部

```
寄存器 (Register File)
  │  延迟: ~0 cycles
  │  容量: 65536 × 32-bit = 256 KB/SM
  │  特点: 线程私有，编译期分配
  ▼
L1 / Shared Memory
  │  延迟: ~30 cycles
  │  容量: 256 KB/SM (Hopper)，可配置 SMEM/L1 比例
  │  特点: Block 内线程共享
  ▼
L2 Cache
  │  延迟: ~200 cycles
  │  容量: 50 MB (H100)，所有 SM 共享
  │  分区: 16-20 个 partition，每 partition 带独立内存控制器
  ▼
HBM (High Bandwidth Memory)
    延迟: ~500+ cycles
    容量: 80 GB (H100)
    带宽: 3.35 TB/s (H100)
```

### L1/SMEM 配置

可以通过 API 动态调整 L1 和 Shared Memory 的比例：

```cpp
// 倾向更多 Shared Memory
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

// 配置全局 L1 缓存偏好
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);  // 倾向 SMEM
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);      // 倾向 L1
```

## 4. Occupancy 与线程调度

### Warp 调度

- 每个 SM 有 4 个 Warp Scheduler
- 每周期每个 scheduler 可发射 1 条指令
- **warp 之间零开销切换**：寄存器文件保存所有 warp 的上下文
- 当一个 warp stall（等待访存），scheduler 立即切换到另一个就绪 warp

### Occupancy 公式

```
Occupancy = min(
    活跃 warp 数 / 64,                    // warp 数量上限
    65536 / (每线程寄存器数 × 32),          // 寄存器上限
    SMEM总量 / (每 block SMEM 用量 × block 数 × 每 block warp 数) // SMEM 上限
)
```

### 典型的 Occupancy 影响因素

| 因素 | 如何影响 | 优化方向 |
|------|---------|---------|
| 寄存器用量 | 每线程寄存器太多 → 活跃 warp 数受限 | 减少 `#pragma unroll`、拆分 kernel |
| Shared Memory | 每 block SMEM 太多 → 活跃 block 数受限 | 减小 tile 尺寸 |
| Block 大小 | Block 太小 → 总 warp 数不足 | 增大 block 维度 |
| Warp stall | 所有 warp 都在等访存 | 提高 occupancy 让更多 warp 可切换 |

## 5. H100 L2 缓存特性

### L2 分区与 SM 关系

H100 有 132 个 SM 共享 50 MB L2。L2 被分为约 16-20 个 partition，**SM 到 L2 partition 的映射不是均匀的**——相邻 SM 可能映射到同一 partition，造成局部竞争。

```
L2 Partition 0 ←── SM 0-7
L2 Partition 1 ←── SM 8-15
...
```

这意味着 **SM 的放置策略会影响 L2 命中率和带宽**：

- 顺序分配 SM（blockIdx → SM）可能导致热点 partition
- 交错分配可分散 L2 访问压力

### 对性能的影响

- 小数据量（完全放 L2）：影响较小
- 中等数据量（部分 L2 命中）：SM 放置策略影响显著
- 大数据量（L2 基本 miss）：主要由 HBM 带宽决定

## 相关资源

- [架构总览](README.md)
- [H100 缓存实验分析](../sm/sm_interval/h100_cache_analysis.md)
- [SM 相关实验代码](../sm/)
