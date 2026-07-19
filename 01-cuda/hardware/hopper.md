# Hopper 架构

Hopper 是 NVIDIA 2022 年发布的架构，代表 GPU 为 **H100、H200**，Compute Capability 9.0。

## 1. 架构亮点

| 特性 | 说明 | 对性能的影响 |
|------|------|------------|
| FP8 Tensor Core | 第四代 TC，FP8 吞吐是 FP16 的 2 倍 | 计算吞吐翻倍 |
| TMA | 硬件 DMA 引擎，异步搬运 Tensor | 释放 SM，隐藏访存延迟 |
| WGMMA | Warp Group (128线程) 级 MMA | Tensor Core 利用率更高 |
| Thread Block Cluster | Block 组成 Cluster，跨 Block 共享数据 | 减少 GMEM 重复读取 |
| DSMEM | 分布式共享内存，Block A 可读 Block B 的 SMEM | Cluster 内高效数据交换 |
| Transformer Engine | 自动 FP8/FP16 精度切换 | 训练/推理精度自适应 |

## 2. TMA（Tensor Memory Accelerator）

### 数据通路

```
Global Memory (DRAM) → L2 Cache → TMA Hardware Engine → Shared Memory (SRAM)
```

### 与 cp.async 的区别

| | cp.async (Ampere) | TMA (Hopper) |
|---|---|---|
| 数据通路 | LD/ST 通路，可能阻塞其他访存 | 独立 DMA 通路，不阻塞 |
| 发起线程 | 需要多个线程协作 | **单个线程**即可发起整个 CTA 的数据请求 |
| 对齐要求 | 4/8/16 字节 | 16 字节 (128b) |
| 越界处理 | 需要手动处理 | **硬件自动补 0**（K 维度越界时） |
| 多维 Tensor | 需要手动计算地址 | 支持多维 Tensor 描述符 |

### 代码入口

- [TMA 基础用法](../hopper/TMA/main.cu)
- [TMA + Pipeline](../hopper/pipe/pipe_tma.cu)
- [TMA 文档](../hopper/TMA/README.md)

## 3. WGMMA（Warp Group MMA）

### 概念

Hopper 的 Tensor Core 以 **Warp Group（4 个 Warp = 128 线程）** 为单位执行矩阵乘法，相比 Ampere 的 warp 级 MMA：

- 更大的计算规模：单条 WGMMA 指令可处理 **64×64×16** 的 tile
- 直接读 Shared Memory：不需要先加载到寄存器，减少寄存器压力
- 异步执行：`warpgroup_arrive()` → `gemm()` → `warpgroup_commit_batch()` → `warpgroup_wait<0>()`

### 数据布局要求

WGMMA 要求矩阵在 Shared Memory 中按 **K-major** 布局（即 K 维连续），需要使用 swizzle 避免 bank conflict：

```cpp
// Shared Memory Layout for WGMMA
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));

// WGMMA Instruction
TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
```

### 代码入口

- [WGMMA 完整 GEMM (TMA + Pipeline + Cluster)](../hopper/wgmma/demo.cu)
- [SM80 MMA vs SM90 WGMMA 对比](../tensorCore/wgmma.cu)

## 4. Thread Block Cluster + DSMEM

### Cluster

通过 `__cluster_dims__(N, 1, 1)` 将相邻的 N 个 Block 组成一个 Cluster，利用 GPC 内的 SM-to-SM 网络进行低延迟通信：

```cpp
__global__ void __cluster_dims__(2, 1, 1) my_kernel() {
    cg::cluster_group cluster = cg::this_cluster();
    // ...
}
```

### 分布式共享内存（DSMEM）

Cluster 内的 Block 可以**直接读写其他 Block 的 Shared Memory**：

```cpp
int *dsmem = cluster.map_shared_rank(&smem_buf, neighbor_rank);
```

关键 API：

| API | 用途 |
|-----|------|
| `cluster.block_rank()` | 获取本 Block 在 Cluster 中的编号 |
| `cluster.map_shared_rank(ptr, rank)` | 获取其他 Block 的 SMEM 地址 |
| `cluster.barrier_arrive()` | 非阻塞同步信号 |
| `cluster.barrier_wait()` | 等待其他 Block 的 arrive 信号 |
| `cluster.sync()` | 全 Cluster 同步 |

### 代码入口

- [DistributedSM / 2-CTA 示例](../hopper/DistributedSM/main.cu) — `__cluster_dims__(2,1,1)` 跨 Block 共享内存归约
- [分布式共享内存文档](../hopper/DistributedSM/README.md)
- [Cooperative Groups 教程](../cg/README.md) — cluster_group 完整 API

## 5. Pipeline（流水线）

Hopper 的 TMA + mbarrier 配合可以实现高效的异步流水线：

```
Pipe 0: [TMA load] ──mbarrier──> [WGMMA compute]
Pipe 1:          [TMA load] ──mbarrier──> [WGMMA compute]
Pipe 2:                   [TMA load] ──mbarrier──> [WGMMA compute]
        时间 ───────────────────────────────────────>
```

核心机制：
- **Producer Barrier**（`ClusterTransactionBarrier`）：TMA 完成后信号通知
- **Consumer Barrier**（`ClusterBarrier`）：WGMMA 消费完成后通知 Producer 可以覆盖
- **Pipeline State**：环形缓冲区 + 相位追踪，管理多级流水

### 代码入口

- [Pipeline 文档](../hopper/pipe/README.md)
- [Pipeline + TMA 示例](../hopper/pipe/pipe_tma.cu)

## 6. H100 缓存层次

```
H100 (132 SMs)
├── 每 SM: L1/Shared Memory 256KB (可配置)
│   └── 典型配置: 128KB SMEM + 128KB L1
├── L2 Cache: 50 MB (所有 SM 共享)
│   └── 分为 16-20 个 partition
└── HBM3: 80 GB, 带宽 3.35 TB/s
```

详见 [SM 微架构](sm.md)

## 相关资源

- [架构总览](README.md)
- [Hopper 代码目录](../hopper/)
