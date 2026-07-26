# TMA（Tensor Memory Accelerator）

TMA 是 Hopper 架构引入的硬件 DMA 引擎，专门负责 Global Memory → Shared Memory 之间的多维 Tensor 异步搬运，**全程不占用 SM 计算单元**。

## 1. 数据通路

```text
Global Memory (DRAM) → L2 Cache → TMA Hardware Engine → Shared Memory (SRAM)
```

TMA 走独立的 DMA 通路，与 SM 的 LD/ST 通路并行，不会阻塞其他访存请求。

## 2. 与 cp.async 的对比

| 维度 | cp.async (Ampere) | TMA (Hopper) |
|------|------------------|-------------|
| 数据通路 | LD/ST 通路，可能阻塞其他访存 | 独立 DMA 通路，不阻塞 |
| 发起线程 | 需要多个线程协作 | **单个线程**即可发起整个 CTA 的数据请求 |
| 对齐要求 | 4/8/16 字节 | 16 字节 (128b) 起始地址，stride 也需对齐 |
| 越界处理 | 需要手动处理 | **硬件自动补 0**（K 维度越界时） |
| 多维 Tensor | 需要手动计算地址 | 支持多维 Tensor 描述符（`cuTensorMap`） |
| 同步机制 | `cp.async.wait_group` | mbarrier（`arrive_and_expect_tx` + `wait`） |

### 为什么 TMA 只需要一个线程发起

cp.async 每个线程拷贝自己负责的数据片段——需要多线程覆盖整个 tile。TMA 把源地址、目的地址、数据量、stride 等**打包成硬件描述符**（`cuTensorMap`），DMA 引擎自动解析并分块搬运，单个线程配置好描述符后整个 CTA 的数据自动到位。

## 3. 核心特性

### 3.1 硬件自动补 0

当 K 维度不是 tile 大小的整数倍时，TMA 硬件自动对越界部分补 0，**不需要**像 cp.async 那样手动处理边界条件或预分配 padding。

```text
原始 K = 100，tile K = 64
Tile 0: K[0:64]   — 正常加载
Tile 1: K[64:100] — TMA 自动补 0 到 64
```

### 3.2 多维 Tensor 描述符

```cpp
// cuTensorMap 描述一个 2D tensor
CUtensorMap tensorMap;
tensorMap.tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
// 设置各维度大小和 stride
cuTensorMapEncodeTiled(&tensorMap, ...);
```

### 3.3 对齐要求

- 起始地址：16 字节对齐
- 各维度 stride：16 字节对齐
- TMA 内部传输单元大小也受对齐约束

## 4. TMA + Pipeline

TMA 与 mbarrier 配合可实现高效的异步流水线：

```text
Pipe 0: [TMA load → SMEM] ──mbarrier──> [compute]
Pipe 1:          [TMA load → SMEM] ──mbarrier──> [compute]
Pipe 2:                   [TMA load → SMEM] ──mbarrier──> [compute]
         时间 ───────────────────────────────────────>
```

关键机制：
- **Producer**（TMA 线程）：`mbarrier.arrive_and_expect_tx(bytes)` → `cp.async.bulk`（即 TMA copy）→ TMA 完成后自动通知 barrier
- **Consumer**（计算线程）：`mbarrier.try_wait` 等待数据就绪

## 5. Blackwell 的变化

Blackwell 架构中 TMA Load 的终点扩展：除了 Shared Memory，还可以写入 **TMem**（Tensor Memory）：

```text
Hopper:    GMEM ──TMA──> SMEM
Blackwell: GMEM ──TMA──> SMEM
           GMEM ──TMA──> TMem (256KB/SM)
```

这使得 B 矩阵等频繁访问的数据可以预取到 TMem 持久化，配合 UMMA 的 TS 模式（A 来自 TMem，B 来自 SMEM）。

## 6. 代码入口

| 文件 | 内容 |
|------|------|
| [hopper/TMA/main.cu](../hopper/TMA/main.cu) | TMA 基础用法 |
| [hopper/TMA/simple.cu](../hopper/TMA/simple.cu) | TMA 简单示例 |
| [hopper/TMA/debug_tma.cu](../hopper/TMA/debug_tma.cu) | TMA 调试代码 |
| [hopper/pipe/pipe_tma.cu](../hopper/pipe/pipe_tma.cu) | TMA + Pipeline 完整实现 |
| [hopper/wgmma/demo.cu](../hopper/wgmma/demo.cu) | WGMMA + TMA + Cluster Launch 的完整 GEMM |
| [cutlass/copy/vec_copy.cu](../cutlass/copy/vec_copy.cu) | CUTLASS 中的 TMA 拷贝 |
| [sm/sm_interval/detailed_analysis.md](../sm/sm_interval/detailed_analysis.md) | SM 实验中涉及 TMA 的 cache 行为分析 |

## 7. 相关文档

- [Hopper 架构专题](../hardware/hopper.md)
- [Blackwell 架构专题](../hardware/blackwell.md)
- [Pipeline 文档](../hopper/pipe/README.md)
- [内存系统](../memory/)
