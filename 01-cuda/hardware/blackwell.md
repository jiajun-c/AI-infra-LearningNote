# Blackwell 架构

Blackwell 是 NVIDIA 2024 年发布的架构，代表 GPU 为 **B200、GB200 (NVL72)**，Compute Capability 10.0/10.1。

## 1. 架构总览

相比 Hopper，Blackwell 的主要升级：

| 维度 | Hopper (H100) | Blackwell (B200) |
|------|-------------|-----------------|
| Tensor Core | 第四代 (FP8) | 第五代 (FP4) |
| Transformer Engine | 第一代 | 第二代 |
| NVLink | NVLink 4.0 | NVLink 5.0 + Switch Chip (NVL72) |
| 内存 | HBM3/HBM3e | HBM3e (8TB/s) |
| 可靠性 | — | RAS 引擎 |

## 2. UMMA

Blackwell 引入了 **UMMA（Unified MMA）**，替代前几代的 MMA 指令。最大特点：**操作数可以来自不同的存储层级**。

### 操作数组合

| 模式 | 矩阵 A | 矩阵 B | 说明 |
|------|-------|-------|------|
| SS | Shared Memory | Shared Memory | 与 Hopper WGMMA 的 SS 模式类似 |
| TS | **TMem** | Shared Memory | 新能力：A 来自 TMem，B 来自 SMEM |

### TMem (Tensor Memory)

- 每个 SM 有 **256KB TMem**
- 位于 SM 内部，延迟接近寄存器
- 专门用于持久化存储 B 矩阵（减少重复从 GMEM 加载）
- 通过 TMA 将数据从 Global Memory 加载到 TMem

```
GMEM ──TMA──> TMem (256KB/SM)
GMEM ──TMA──> Shared Memory
```

## 3. 双 SM 协同处理

Blackwell 支持 **两个 SM 协同处理一条 UMMA 指令**：

```
LeaderCTA (SM_A)          PeerCTA (SM_B)
      │                        │
      ├── 执行 MMA 指令 ◄──────┤ 提供数据
      │                        │
      ├── B 矩阵在本 SM 的 SMEM ─┤ B 矩阵在对方 SM 的 SMEM
      │   (N/2 列)              │   (N/2 列)
```

- **数据分区**：B 矩阵在两个 SM 间分区，每个 SMEM 中 B 的维度 = N / 2
- **更大 tile**：单次 MMA 可处理 **256×64×32** 的 tile
- **通信机制**：通过 SM-to-SM 网络（DSMEM 机制）协同

## 4. TMA 的变化

在 Hopper 中 TMA Load 的终点只有 Shared Memory：

```
Hopper:  GMEM ──TMA──> SMEM
```

Blackwell 中增加了 **TMem** 作为终点：

```
Blackwell:  GMEM ──TMA──> SMEM
            GMEM ──TMA──> TMem
```

这意味着可以将频繁访问的数据（如 B 矩阵）通过 TMA 预取到 TMem 中持久化，减少重复加载。

## 5. 其他新特性

### NVLink Switch Chip (NVL72)

- 通过 NVLink Switch Chip 实现 **72 GPU 全互联**
- 多机通信延迟降低到机内 NVLink 水平
- 对 MoE / TP 等通信密集型场景收益巨大

### RAS 引擎

- **R**eliability, **A**vailability, **S**erviceability
- 硬件级的故障检测和恢复
- 针对万卡级集群的可靠性需求

### 解压缩引擎

- 硬件加速数据解压
- 解决大规模训练中的 I/O 瓶颈

## 相关资源

- [架构总览](README.md)
- [代码示例](../blackwell/)（更多示例待补充）
