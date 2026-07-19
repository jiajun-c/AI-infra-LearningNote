# GPU 架构

## 阅读顺序

1. **物理层级** — 理解 GPU 的硬件组织（GPC → TPC → SM）
2. **历代架构迭代** — 掌握从 Volta 到 Blackwell 的关键特性演进
3. **各架构专题** — 深入特定架构：[Hopper](hopper.md) | [Blackwell](blackwell.md)
4. **SM 微架构** — 深入 SM 内部：[SM 微架构](sm.md)

代码示例目录：[01-cuda/hopper/](../hopper/) | [01-cuda/blackwell/](../blackwell/) | [01-cuda/ampere/](../ampere/)

---

## 1. GPU 物理层级

GPU 的物理组织分为三级：

```
GPC (图形处理集群)
├── TPC (纹理处理集群) × N
│   ├── SM (流式多处理器) × 2  (Hopper 中 1 TPC = 2 SM)
│   └── SM
├── TPC
│   └── ...
└── ...
```

### GPC (Graphics Processing Cluster)

- GPU 内部最大的功能单元块
- Hopper 架构引入的 **Thread Block Cluster** 在硬件上对应于 GPC
- GPC 内部包含专用的 **SM-to-SM 网络**，允许 GPC 内的 SM 之间低延迟通信（DSMEM）

### TPC (Texture Processing Cluster)

- 嵌套在 GPC 内部
- Hopper 架构中 **1 TPC = 2 SM**
- TPC 是硬件屏蔽（Floorsweeping）和掩码控制的物理单位

### SM (Streaming Multiprocessor)

- GPU 执行计算任务的基本单元
- 每个 SM 内部进一步分为 **4 个 Subpartition**（子分区/处理块），每个 subpartition 包含：
  - **1 个 Warp Scheduler** — 每周期从就绪 warp 中选择一条指令发射
  - **1 个 Dispatch Unit** — 将指令分发到执行单元
  - **CUDA Cores** — FP32/INT32 运算单元（Hopper 每 subpartition 含 16 条 FP32 + 16 条 INT32）
  - **Tensor Cores** — 矩阵乘加加速器（Hopper 每 subpartition 含 1 个）
  - **LD/ST 单元** — 全局/局部内存访问
  - **寄存器堆（Register File）** — 整个 SM 共 65536 个 32-bit 寄存器，按 subpartition 分区
- SM 级别共享资源：
  - **L1 Cache / Shared Memory**（Hopper 为 256KB，可配置比例）
  - **L0 指令缓存**
  - **Constant Cache**

```text
SM (Streaming Multiprocessor)
├── Subpartition 0
│   ├── Warp Scheduler + Dispatch Unit
│   ├── CUDA Cores (FP32 × 16 + INT32 × 16)
│   ├── Tensor Core × 1
│   ├── Register File (65K / 4)
│   └── LD/ST Unit
├── Subpartition 1
│   └── ...（同 Subpartition 0）
├── Subpartition 2
│   └── ...
├── Subpartition 3
│   └── ...
├── L1 Cache / Shared Memory (256 KB，4 个 subpartition 共享)
└── L0 Instruction Cache

---

## 2. 历代架构迭代

### 2.1 Volta（2017）— Tensor Core 元年

代表 GPU：**V100**

| 类别 | 特性 |
|------|------|
| 硬件 | 第一代 Tensor Core |
| 硬件 | HBM2 显存，带宽 900 GB/s |
| 硬件 | NVLink 2.0，提升 GPU 互联带宽 |
| 软件 | `nvcuda::wmma` 命名空间，直接调用 Tensor Core 做 16×16×16 矩阵块计算 |
| 软件 | 独立线程调度（Independent Thread Scheduling） |

### 2.2 Turing（2018）— RT Core + INT8

代表 GPU：**T4, RTX 20 系列**

| 类别 | 特性 |
|------|------|
| 硬件 | **RT Core**：硬件加速 BVH 遍历（光线追踪） |
| 硬件 | 第二代 Tensor Core：新增 **INT8、INT4** 支持 |
| 硬件 | **统一 L1/Shared Memory**：L1 和 Shared Memory 合并为一块物理 SRAM，可动态配置比例 |
| 软件 | CUDA 10 |

### 2.3 Ampere（2020）— TF32 + 异步拷贝

代表 GPU：**A100, RTX 30 系列**

| 类别 | 特性 |
|------|------|
| 硬件 | 第三代 Tensor Core：支持 **TF32**（FP32 范围 + FP16 精度）、**BF16**、**结构化稀疏**（2:4 模式，吞吐翻倍） |
| 硬件 | **异步拷贝引擎**：Global Memory → Shared Memory 直连通路，绕过寄存器 |
| 软件 | **cp.async**：从 Global Memory 异步加载数据到 Shared Memory，减少寄存器压力、隐藏延迟 |
| 软件 | CUDA 11 |

### 2.4 Hopper（2022）— FP8 + TMA + WGMMA

代表 GPU：**H100, H200**

| 类别 | 特性 |
|------|------|
| 硬件 | 第四代 Tensor Core：支持 **FP8**（E4M3, E5M2），吞吐是 FP16 的 2 倍 |
| 硬件 | **TMA**（Tensor Memory Accelerator）：可编程异步 DMA 引擎，负责 Global→Shared 的复杂 Tensor 搬运，**不占用 SM** |
| 硬件 | **Thread Block Cluster**：Block 可在 GPC 内协同，支持 **DSMEM**（跨 Block 访问 Shared Memory） |
| 硬件 | **Transformer Engine**：软硬件结合，自动管理 FP8 精度切换 |
| 软件 | **WGMMA**（Warp Group MMA）：以 Warp Group（4 Warp = 128 线程）为单位计算，直接读 Shared Memory |
| 软件 | CUDA 12 |

详见 [Hopper 架构专题](hopper.md)

### 2.5 Blackwell（2024）— FP4 + UMMA

代表 GPU：**B200, GB200 (NVL72)**

| 类别 | 特性 |
|------|------|
| 硬件 | 第二代 Transformer Engine：支持 **FP4**（微缩放格式），吞吐是 H100 的 2 倍 |
| 硬件 | **NVLink Switch Chip (NVL72)**：72 GPU 全互联，多机延迟降到机内水平 |
| 硬件 | **RAS 引擎**：硬件级可靠性、可用性、可维护性，用于超大规模集群 |
| 硬件 | **解压缩引擎**：硬件加速数据解压，解决 I/O 瓶颈 |
| 软件 | **UMMA**：替代前代 MMA，操作数可来自不同存储层级（SS/TS），支持 TMem |
| 软件 | TMA Load 终点扩展：除了 Shared Memory，还可以到 **TMem** |

详见 [Blackwell 架构专题](blackwell.md)

---

## 3. 关键概念速查

| 概念 | 首次引入 | 简介 | 详见 |
|------|---------|------|------|
| Tensor Core | Volta | 矩阵运算硬件单元 | — |
| cp.async | Ampere | GMEM→SMEM 异步拷贝 | [ampere/](../ampere/cpasync/) |
| TMA | Hopper | 硬件 DMA 引擎，异步搬运 Tensor | [hopper/](../hopper/TMA/) |
| WGMMA | Hopper | Warp Group 级矩阵乘法 | [hopper/](../hopper/wgmma/) |
| DSMEM | Hopper | 跨 Block 共享内存访问 | [hopper/](../hopper/DistributedSM/) |
| UMMA | Blackwell | 统一 MMA，操作数多源 | [blackwell.md](blackwell.md) |
| Cooperative Groups | CUDA 9 | 灵活的线程协作抽象 | [cg/](../cg/) |

---

## 4. SM 微架构

SM 是 GPU 执行计算的基本单元，理解其内部结构对性能优化至关重要。

详见 [SM 微架构](sm.md)

## 5. 相关资源

- [历代架构迭代总览](../../07-system/gpu/README.md) — 系统视角的架构补充
- [CUDA 编程笔记](../)
- [性能分析](../../09-profile/cuda/)
