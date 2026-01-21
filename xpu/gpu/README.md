# GPU 各代的架构迭代

## 1. Volta 架构（2017）

硬件特性
- Volta架构在2017年提出，其引入了第一代Tensor core
- HBM2显存：带宽达到900GB/s
- NVLink2.0 提升了GPU互联带宽

软件特性 (CUDA9.0+)
- 引入了nvcuda::wmma命名空间
- 允许程序员直接调用Tensor core进行16x16x16的矩阵块计算

独立线程调度

## 2. Turing架构（2018）

硬件特性
- RT Core（光线追踪核心）：硬件加速BVH遍历
- Tensor Core（第二代）
    - 新增了INT8和INT4的支持
- 统一L1/Shared Memory: 将L1和shared memory合并为一块物理SRAM，允许根据Kernel动态调整配置

## 3. Ampere架构（2020）

硬件特性
- Tensor core（第三代）
    - TF32一种新的数据格式，保留了TF32的数据范围和FP16的精度
    - BF16：原生支持BFloat16，
    - 结构化稀疏：硬件支持2：4稀疏性，吞吐量翻倍
- 异步硬件拷贝：在L2和shared memory之间建立了直连的数据通路，绕过了寄存器文件

软件特性(cuda11+)
- cp.async: 支持从Global Memory异步加载数据到shared memory
- 优势：极大地减少了寄存器压力，并且可以完美掩盖Gm的延迟

## 4. Hopper架构（2022）

硬件特性
- Transformer Engine: 结合软件栈，自动管理 FP8 精度切换。
- Tensor Core (第四代):
    - FP8: 支持 FP8 (E4M3, E5M2) 格式，吞吐量是 FP16 的两倍。
- TMA（Tensor memory accelerator）
    - 一个可编程的异步DMA引擎
    - 负责从全局内存到共享内存之间的复杂Tensor搬运
    - 解放SM：搬运过程完全不占用SM的CUDA Core
- Thread Block Cluster (线程块集群):
    - 允许一组 Block (Cluster) 在 GPC (Graphics Processing Cluster) 内部协同工作。
    - 支持 Distributed Shared Memory (DSMEM)：Block A 可以直接读写 Block B 的 Shared Memory。

软件特性：
    - WGMMA (Warp Group MMA):
    - 利用 Tensor Core 的新指令，允许多个 Warp (Warp Group) 协同计算。

特性: 可以直接读取 Shared Memory 或 Distributed Shared Memory 作为输入，无需先加载到寄存器。

## 5. BlackWell架构(2024)

代表芯片B200 (GB200)

硬件特性：
    - 第二代 Transformer Engine:
    - FP4: 支持 4-bit 浮点推理。通过微缩放格式 (Micro-scaling) 保持高精度，吞吐量是 H100 的两倍。
    - NVLink Switch Chip: 支持 72 个 GPU 全互联 (NVL72)，将多机通信延迟降低到机内水平。
    - RAS 引擎: 硬件级的可靠性、可用性和可维护性增强，用于超大规模集群的故障检测。
    - 解压缩引擎 (Decompression Engine): 硬件加速数据解压，解决 I/O 瓶颈。