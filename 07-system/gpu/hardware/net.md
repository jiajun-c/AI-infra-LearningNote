# GPU网络拓扑

`nvidia-smi topo -m` 是看一张机器上各 GPU 之间互联关系的最常用入口。它本质上回答两个问题：

- 任意两张 GPU 之间走的是 NVLink 还是 PCIe？
- 走 NVLink 的话，几代、几链路？

下面给一份 DGX H100（8 卡 H100 + 4 颗 NVSwitch3）的典型输出做样本。

## 1. 模拟输出

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    CPU Affinity    NUMA Affinity    NIC Affinity
GPU0     X      NV4     NV4     NV4     NV4     NV4     NV4     NV4     PIX     SYS     0-23            0                0,1
GPU1    NV4     X      NV4     NV4     NV4     NV4     NV4     NV4     PIX     SYS     0-23            0                0,1
GPU2    NV4    NV4      X      NV4     NV4     NV4     NV4     NV4     SYS     PIX     24-47           1                2,3
GPU3    NV4    NV4     NV4      X      NV4     NV4     NV4     NV4     SYS     PIX     24-47           1                2,3
GPU4    NV4    NV4     NV4     NV4      X      NV4     NV4     NV4     PIX     SYS     48-71           2                4,5
GPU5    NV4    NV4     NV4     NV4     NV4      X      NV4     NV4     PIX     SYS     48-71           2                4,5
GPU6    NV4    NV4     NV4     NV4     NV4     NV4      X      NV4     SYS     PIX     72-95           3                6,7
GPU7    NV4    NV4     NV4     NV4     NV4     NV4     NV4      X      SYS     PIX     72-95           3                6,7
NIC0    PIX    PIX     SYS     SYS     PIX     PIX     SYS     SYS      X      SYS
NIC1    SYS    SYS     PIX     PIX     SYS     SYS     PIX     PIX     SYS      X

Legend:

  X    = Self
  NV4  = Connection two adjacent GPUs via NVLink with 4 links
  NV6  = Connection two adjacent GPUs via NVLink with 6 links
  PIX  = Connection traversing at most a single PCIe host bridge
  PXB  = Connection traversing multiple PCIe host bridges without crossing NUMA nodes
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
  SYS  = Connection traversing PCIe as well as SMP interconnect between NUMA nodes (e.g. QPI/UPI)
  NODE = Connection traversing PCIe as well as SMP interconnect between NUMA nodes (e.g. QPI/UPI)
  SOC  = Connection traversing on-chip SoC interconnect (e.g. NVLink-C2C)
```

## 2. 怎么读这张矩阵

矩阵是**对称**的，第 `(i, j)` 个格表示 **第 i 个设备到第 j 个设备**之间走的是什么通路。这里的"设备"既包括 GPU，也包括 NIC（NIC0、NIC1），所以矩阵右下角会出现 NIC ↔ NIC、NIC ↔ GPU 这两段。

矩阵可以拆成三块来看：

```text
          GPU 列           NIC 列
       ┌──────────────┐ ┌──────────────┐
GPU 行 │  NVLink / PCIe│ │  PIX / SYS   │   ← GPU ↔ GPU
       ├──────────────┤ ├──────────────┤
NIC 行 │  PIX / SYS   │ │  PIX / SYS   │   ← GPU ↔ NIC, NIC ↔ NIC
       └──────────────┘ └──────────────┘
```

- **GPU ↔ GPU**：左上角，要么是 NVLink（NV*），要么是 PCIe（PIX/PXB/PHB/SYS）。
- **GPU ↔ NIC**：右上角和左下角，决定**GPUDirect RDMA**走的是直连 PIX 还是跨 NUMA 的 SYS。这部分就是分布式训练最关心的，**NIC Affinity** 那一列正是从这里算出来的。
- **NIC ↔ NIC**：右下角，本机内 NIC 通常走 PIX（同一 PCIe switch）或 SYS（跨 NUMA）。

- **X**：自己到自己，没意义。
- **NV4 / NV6**：两张 GPU 通过 NVLink 直连或经 NVSwitch 中转，数字表示有几条 link（NVLink 4 / NVLink 6）。这是机内 GPU 通信的"高速通路"，H100 单条 NVLink 4 大约 25 GB/s 单向带宽。
- **PIX**：走 PCIe，且只跨**一个** PCIe host bridge。比 NVLink 慢，但比下面几种都快（PCIe Gen5 x16 ≈ 64 GB/s 双向，但实际单向要再降）。
- **PXB**：走 PCIe，跨**多个** PCIe switch，但仍在一个 NUMA node。
- **SYS / NODE**：跨 NUMA 节点，要再过 CPU 的片间总线（QPI / UPI / xGMI 等），延迟和带宽都差很多，**机内**通信里看到 SYS / NODE 是非常影响 NCCL 性能的信号。
- **SOC**：SoC 片内互联，主要出现在 GH200 这种 CPU-GPU 共享封装（NVLink-C2C）的形态上。

最右三列是 **CPU Affinity**、**NUMA Affinity** 和 **NIC Affinity**：

- **CPU Affinity**：这张 GPU 物理上挂在哪几个 CPU 核的 PCIe root complex 下，做 NUMA-aware 绑核时用。
- **NUMA Affinity**：这张 GPU 在哪个 NUMA node 上，跨 NUMA 通信要走 QPI/UPI，对 all-to-all 类 collective 影响很大。
- **NIC Affinity**：这张 GPU 物理上"靠近"哪些网卡（`mlx5_0`/`mlx5_1` 这类）。这里的"近"指 GPU 和 NIC 在**同一颗 PCIe switch / 同一 NUMA 节点 / 走 GPUDirect RDMA 通路**。这一列对分布式训练很关键，见下一节。

## 3. DGX H100 上的物理拓扑对照

把上面的矩阵和硬件对上，能加深理解：

```text
        ┌─────────────────────────────────────────┐
        │              DGX H100 baseboard         │
        │                                         │
        │   GPU0 GPU1     GPU2 GPU3     GPU4 GPU5  GPU6 GPU7
        │   (NUMA 0)      (NUMA 1)     (NUMA 2)   (NUMA 3)
        │     │  │          │  │          │  │      │  │
        │     └──┴────┐  ┌──┴──┘      ┌──┴──┘  ┌──┴──┘
        │             │  │            │        │
        │        ┌────┴──┴────┐  ┌────┴────────┴────┐
        │        │  NVSwitch0 │  │  NVSwitch1 ...   │
        │        └────┬──┬────┘  └────┬────────┬────┘
        │             │  │            │        │
        │             └──┴──┐  ┌──────┴────────┘
        │                    │  │
        │        ... NVSwitch2 / NVSwitch3 ...
        │                                         │
        └─────────────────────────────────────────┘
```

关键事实：

- 每张 H100 通过 **18 条 NVLink** 接到 4 颗 NVSwitch，每条约 25 GB/s 单向，所以每张 GPU 的总 NVLink 出口带宽 ≈ 450 GB/s 单向 / 900 GB/s 双向。
- 任意两张 GPU 之间的 traffic，**理论都可以走 NVSwitch 全带宽**（即矩阵里全是 NV4，不是 SYS）。但实际 NCCL ring / tree 算法能不能充分利用，取决于它把哪些 GPU 排在 ring 同侧。
- 矩阵里 NVLink 链路数（NV4 表示 4 条 link）只反映"基线能力"，真实流量会被 NVSwitch 的 crossbar 调度约束。

## 4. 和 PCIe-only 拓扑对比

如果机器上**没有** NVSwitch，只靠 PCIe 互联，输出会变成下面这样（典型 8 卡 PCIe-only 服务器）：

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    CPU Affinity    NUMA Affinity    NIC Affinity
GPU0     X      PIX     PIX     PIX     SYS     SYS     SYS     SYS     PIX     SYS     0-23            0                0,1
GPU1    PIX     X      PIX     PIX     SYS     SYS     SYS     SYS     PIX     SYS     0-23            0                0,1
GPU2    PIX    PIX      X      PIX     SYS     SYS     SYS     SYS     PIX     SYS     24-47           1                0,1
GPU3    PIX    PIX     PIX      X      SYS     SYS     SYS     SYS     PIX     SYS     24-47           1                0,1
GPU4    SYS    SYS     SYS     SYS      X      PIX     PIX     PIX     SYS     PIX     48-71           2                2,3
GPU5    SYS    SYS     SYS     SYS     PIX      X      PIX     PIX     SYS     PIX     48-71           2                2,3
GPU6    SYS    SYS     SYS     SYS     PIX     PIX      X      PIX     SYS     PIX     72-95           3                2,3
GPU7    SYS    SYS     SYS     SYS     PIX     PIX     PIX      X      SYS     PIX     72-95           3                2,3
NIC0    PIX    PIX     PIX     PIX     SYS     SYS     SYS     SYS      X      SYS
NIC1    SYS    SYS     SYS     SYS     PIX     PIX     PIX     PIX     SYS      X
```

注意区别：

- 同一 NUMA node 内：PIX（PCIe 直连 / 共享 switch）。
- 跨 NUMA node：**SYS**，必须再过 CPU 总线。
- **完全没有 NV*** 这一行。

这种拓扑下，NCCL AllReduce 走 PCIe 跨 NUMA 的部分会变成瓶颈，DP 训练很容易在跨 NUMA 通信上耗时翻倍。

## 5. 为什么 NIC Affinity 这一列很重要

`nvidia-smi topo -m` 的右栏里，**最容易被人忽略的就是 NIC Affinity**，但它对分布式训练性能影响最大。原因是从 GPU 出去的"远端通信"——也就是 DP / TP 跨节点的 collective——实际走的路径是：

```text
GPU HBM
   ↓ PCIe P2P (GPUDirect RDMA)
本机 NIC
   ↓ 网络 (IB NDR / RoCE v2 / IB HDR)
对端 NIC
   ↓ PCIe P2P
对端 GPU HBM
```

这条路径里**最慢的一段不是网络，而是 GPU ↔ NIC**。如果一张 GPU 用的不是它"近"的 NIC，traffic 就要跨 NUMA 走一次（也就是矩阵里看到的 SYS 路径），这一步会直接拖垮 cross-node 通信。

具体来说：

| 场景 | GPU ↔ NIC 路径 | 后果 |
| --- | --- | --- |
| GPU 与 NIC 同 NUMA、同 PCIe switch | PIX / NVLink | GPUDirect RDMA 直连，最快 |
| GPU 与 NIC 跨 NUMA | **SYS** | GPUDirect RDMA 仍可工作，但需要先绕一圈 CPU 总线，带宽和延迟都差几倍 |
| GPU 与 NIC 不支持 P2P（极少数情况） | 不通 | 回落到 staged copy，巨慢 |

因此分布式训练框架在初始化时，会按 NIC Affinity 给 rank 编号，让**编号相邻的 rank 落在 NIC Affinity 相邻的 GPU 上**，这样 cross-node ring 的每一跳都能落到 GPUDirect RDMA 直连路径上。

几个常见的运维动作：

```bash
# 1) 查所有 IB 网卡的 GPU 亲和性（更细的版本）
nvidia-smi topo -p2p r

# 2) 强制 NCCL 只用某些 NIC（避免它选到错的）
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

# 3) 强制 CPU 亲和性绑核（让发包线程靠近对应的 NIC 和 GPU）
numactl --cpunodebind=0 --membind=0 python train.py
```

## 6. 多节点情况

`nvidia-smi topo -m` 只看**单节点内**的拓扑。跨节点一般用 `nvidia-smi topo -p2p r` 或者直接看 NCCL / IB 子网配置（`show_ib`）：

- 多节点之间一般是 InfiniBand 或 RoCE，组成 Fat-Tree / Dragonfly / Torus。
- NCCL 会根据物理拓扑自动选 ring / tree / rail-optimized 路径，所以集群配置一般会做 `NCCL_TOPO_FILE` / `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` 等调优。

机间拓扑细节见 [04-comm/topo/README.md](../../04-comm/topo/README.md)。
