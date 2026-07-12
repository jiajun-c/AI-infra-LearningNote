# GPU-initiated Networking (GIN)

## 1. 背景

GPU-initiated Networking 是 NCCL 提供的一种高级特性，允许 **GPU Kernel 在设备端直接发起网络通信**，无需 CPU 介入。

### 传统 RDMA vs GIN

```text
传统 RDMA 数据路径:
  GPU Kernel 完成计算
    → CPU: 收到完成通知
      → CPU: 调用 ibv_post_send() 提交 WR
        → NIC: 处理 WQE，DMA 读取 GPU 显存
          → 网络发送

GIN 数据路径:
  GPU Kernel 完成计算
    → GPU Kernel 直接写 NIC 的门铃寄存器 (doorbell)
      → NIC: 处理 WQE，DMA 读取 GPU 显存
        → 网络发送
```

关键区别：GIN 消除了 CPU 这个中间人，GPU Kernel 直接操控网卡。

---

## 2. 使用场景

- 多节点，跨主机通信
- 设备端发起的网络通信与异步完成通知
- 与 LSA (Load Store Access) 组合的混合拓扑优化：本地（节点内）使用 LSA（NVLink 直接 load/store 远端显存），远端（跨节点）使用 GIN
- 对称内存窗口下的高效集合通信 (All2All, AllGather, ReduceScatter)
- **Kernel Fusion**：GPU Kernel 计算完成后立即通过 GIN 发送结果，无需切回 CPU 提交 WR，减少 kernel launch 延迟

---

## 3. 核心机制详解

### 3.1 门铃 (Doorbell) 机制

RDMA 网卡通过 **MMIO (Memory-Mapped I/O)** 暴露寄存器到主机/GPU 的地址空间。CPU 或 GPU 向特定地址写特定值，就能触发网卡动作。

```text
NIC 的 MMIO 空间映射到 GPU 地址空间（BAR1）:

  GPU 虚拟地址空间
  ┌────────────────────┐
  │  GPU 显存 (HBM)     │
  │                    │
  ├────────────────────┤
  │  NIC MMIO 映射     │  ← GPU Kernel 可直接 load/store 这些地址
  │  ├─ SQ 门铃寄存器   │     写 "WQE数量" → 网卡收走 SQ 中的新 WQE
  │  ├─ CQ 头指针       │     读 CQ 完成通知
  │  └─ 其他寄存器      │
  └────────────────────┘
  
```

GPU Kernel 发数据的过程：

1. GPU Kernel 写好数据到显存
2. GPU Kernel 写 WQE 到显存的 SQ Buffer（NIC 可 DMA 读的区域）
3. GPU Kernel 向 NIC 的门铃寄存器写入 "1"（表示新提交了 1 个 WQE）
4. NIC 读取 SQ 中的 WQE，执行 DMA，发送数据

### 3.2 为什么 GIN 可以跨机？

这个问题看似矛盾——"GPU 不是一个计算芯片吗，怎么能跨网络通信？"

核心答案：**GPU 只负责"发号施令"（按门铃），真正跨机传输数据的是网卡**。

```text
┌──────────────────────────────────────────────────────────────┐
│                        GIN 的职责边界                          │
│                                                              │
│   GPU 做的事（GIN 的范围）:                                    │
│    1. 准备好数据在显存                                         │
│    2. 写好 WQE（告诉网卡：从哪儿取、取多长、发给谁）              │
│    3. 按 NIC 的门铃 ← 就这一步是 GIN 的核心创新                  │
│                                                              │
│   NIC 做的事（和传统 RDMA 完全一样）:                            │
│    1. 收到门铃，读取 SQ 中的 WQE                                │
│    2. 通过 GPUDirect RDMA，DMA 读取 GPU 显存数据                 │
│    3. 封装成网络包，经 InfiniBand/RoCE 发出                       │
│    4. 远端 NIC 收到，写入远端 GPU 显存                           │
│                                                              │
│  结论: 跨机能力来自 NIC，不来自 GPU。                            │
│        GIN 只是把"谁按门铃"从 CPU 换成了 GPU。                    │
└──────────────────────────────────────────────────────────────┘
```

**三个硬件能力缺一不可**：

| 能力 | 硬件 | 说明 | 从哪代开始支持 |
| ---- | ---- | ---- | -------------- |
| **GPU 访问 NIC MMIO** | GPU BAR1 映射 | GPU 可以向 NIC 的控制寄存器发 load/store，这是"按门铃"的前提 | Hopper (H100) |
| **NIC 访问 GPU 显存** | GPUDirect RDMA | NIC 可以直接 DMA 读写 GPU HBM，无需 CPU 内存中转 | Kepler + CX-3 |
| **NIC 跨机传输** | InfiniBand / RoCE | NIC 自身的网络能力，封装、路由、重传 | CX-3+ |

三者配合：

```text
GPU0 (Hopper)                              GPU1 (Hopper)
  │                                           ▲
  │ ①GPU store 到 NIC MMIO (按门铃)            │
  ▼                                           │
NIC0 (CX-7) ──── InfiniBand/RoCE ────→ NIC1 (CX-7)
  │                                           │
  │ ②NIC DMA 读 GPU0 显存                      │ ③NIC RDMA WRITE 写 GPU1 显存
  ▼                                           │
GPU0 HBM                                  GPU1 HBM
```

- 步骤 ①：GPU→NIC，走 PCIe MMIO（GIN 的贡献）
- 步骤 ②：NIC→GPU，走 GPUDirect RDMA（早就有）
- 步骤 ③：NIC→NIC，走网络（早就有）

**Hopper 之前为什么不行？** GPU 虽然支持 GPUDirect RDMA（NIC 可以拉取 GPU 显存），但 GPU 不能主动向 NIC 发 MMIO 请求。传统的 PCIe 拓扑中，MMIO 由 Root Complex 仲裁，GPU 只能作为被访问方。Hopper 的 NVLink-C2C + 增强 BAR1 让 GPU 也成为 MMIO 的发起方。

**一个类比**：

> 传统 RDMA 像是：老板（CPU）给快递员（NIC）打电话，说"去仓库（GPU 显存）取个包裹送到 X 地址"。
>
> GIN 像是：仓库管理员（GPU Kernel）自己拿起了电话，直接告诉快递员同样的指令。
>
> 快递员（NIC）还是那个快递员，送包裹的方式没有变。变的只是**谁打的电话**。

### 3.3 与传统 RDMA Post 的对比

| | 传统 RDMA (ibv_post_send) | GIN |
| --- | --- | --- |
| **提交者** | CPU | GPU Kernel |
| **门铃写入** | CPU 写 NIC MMIO | GPU Kernel 写 NIC MMIO |
| **WQE 所在位置** | 主机内存 (系统 DRAM) | GPU 显存 (HBM) 或主机内存 |
| **完成通知** | CPU poll CQ | GPU Kernel 可直接读 CQ（通过 NIC MMIO 映射） |
| **延迟** | CPU intervene ~数 μs | GPU→NIC 直接 ~亚 μs |
| **CPU 占用** | 需要 CPU 轮询或中断 | CPU 完全不受影响 |
| **适用 GPU** | 所有 | H100+ (Hopper 架构及以上) |

---

## 4. 关键组件

### 4.1 LSA (Load Store Access)

LSA 是 GIN 的**节点内**对应物：

```text
┌──────────────────────────────────────────────────────┐
│  单节点 8×GPU (NVSwitch)                             │
│                                                      │
│   GPU0 ──load──→ GPU1 显存  (通过 NVLink C2C)       │
│   GPU0 ←──store── GPU1 显存                          │
│                                                      │
│  LSA: GPU 可以直接 load/store 其他 GPU 的显存        │
│       不需要 RDMA，不需要 QP，纯 NVLink 硬件             │
└──────────────────────────────────────────────────────┘
```

LSA vs GIN：

| | LSA | GIN |
| --- | --- | --- |
| **范围** | 节点内 (intra-node) | 跨节点 (inter-node) |
| **硬件通道** | NVLink / NVSwitch | InfiniBand / RoCE |
| **编程模型** | 直接 load/store | 门铃 + WQE（类似 RDMA） |
| **延迟** | ~几百 ns | ~μs |
| **QP** | 无 | 有 |
| **GPU 要求** | H100+ | H100+ |

### 4.2 对称内存窗口 (Symmetric Memory Windows)

在 GIN 场景中，所有 Rank 约定好在显存中开辟**同样大小、同样偏移**的窗口区域，作为远端访问的目标：

```text
每个 GPU 显存布局:
┌─────────────────────┐  高地址
│  计算数据区域        │
├─────────────────────┤
│  对称窗口 (window)   │  ← 所有 Rank 在此区域的偏移/大小一致
│  远端可直接读写       │     Rank0 可以直接写 Rank1-7 的这段区域
├─────────────────────┤
│  其他                │
└─────────────────────┘  低地址
```

这样 GPU Kernel 只需知道 `base + rank * offset` 就能算出任意远端窗口的地址，配合 GIN 或 LSA 直接访问。

---

## 5. 后端实现机制

### 5.1 GDAKI (GPU Direct Async Kernel-Initiated) 后端

GDAKI 是 NVIDIA 提供的一组 CUDA API，使 GPU Kernel 能够直接操作网卡。核心 API 包括：

```c
// 从 GPU 侧获取网络设备句柄
cudaGetDeviceByNet(...)

// GPU Kernel 中可用的内联操作
// 写门铃: GPU Kernel 直接 store 到网卡的 doorbell 地址
// 读 CQ:  GPU Kernel 直接 load 网卡的 CQ 地址
```

GDAKI 的工作流：

```text
初始化阶段 (CPU 侧):
  1. 创建标准 RDMA 资源 (PD, QP, CQ, MR)
  2. 将 QP 的 SQ/RQ Buffer 分配在 GPU 可访问的内存 (GPU BAR1 或显存)
  3. 将网卡的 doorbell MMIO 地址映射到 GPU 地址空间
  4. 将这些地址传递给 GPU Kernel

运行时 (GPU Kernel 侧):
  1. GPU Kernel 写好自己的数据到显存
  2. GPU Kernel 构造 WQE，写入 SQ Buffer
  3. GPU Kernel 向 NIC doorbell 写 1（提交 WQE）
  4. （可选）GPU Kernel 读取 CQ 获取完成通知
```

### 5.2 libnvls.so — NCCL NVLink SHARP

`libnvls.so` 是 NCCL 中实现 LSA 的库，与 GIN 配合使用：

- **节点内**：`libnvls` 通过 NVLink 直接 load/store 远端显存
- **跨节点**：`libnvls` → GIN 通过 RDMA 网卡收发

两者接口统一，上层 NCCL 算法无需区分本地/远端通信路径。

### 5.3 NCCL 中的初始化流程

```text
1. ncclInit() 
   ├─ 检测是否支持 GIN (Hopper GPU + CX-7+)
   ├─ 创建 RDMA 资源 (QP, CQ, MR)
   ├─ 映射 NIC MMIO 到 GPU 地址空间
   └─ 加载 libnvls.so

2. ncclGroupStart() 
   └─ 构建通信拓扑 (LSA 图 + GIN 图)

3. ncclSend/ncclRecv()
   ├─ 本地: libnvls 直接 load/store 远端显存 (LSA)
   └─ 远端: GIN — GPU Kernel 写 doorbell 触发 NIC 发送
```

---

## 6. 完整数据流

### 6.1 节点内 (LSA) — 无 RDMA

```text
GPU0 Kernel 发送:
  1. GPU0 Kernel 计算完成，数据在 HBM
  2. GPU0 Kernel 通过 NVLink 直接 store 到 GPU1 的对称窗口
     （这是一条普通的 GPU store 指令，NVLink 硬件自动路由到远端）
  3. GPU0 写一个 flag 到 GPU1 的窗口，通知数据已就绪

GPU1 Kernel 接收:
  1. GPU1 Kernel 轮询 flag（或等 semaphore）
  2. GPU1 Kernel 通过 NVLink 直接 load GPU0 的数据
     或 GPU0 已经 store 过来了，直接读本地显存即可
```

### 6.2 跨节点 (GIN) — GPU 直接操控网卡

```text
GPU0 Kernel 发送 (GIN):
  1. GPU0 Kernel 计算完成，数据在 HBM
  2. GPU0 Kernel 写 WQE 到 SQ Buffer (映射到 GPU 显存/系统内存)
     WQE 描述:
       - 源地址: GPU0 HBM 中的数据地址 (lkey)
       - 目标地址: GPU1 对称窗口地址 (rkey)
       - 长度: N 字节
  3. GPU0 Kernel 写 NIC doorbell: store 1 → NIC MMIO 门铃地址
  4. NIC 收到 doorbell → 读 SQ 中的 WQE → DMA 读 GPU0 HBM → 网络发送
  5. （可选）GPU0 Kernel 轮询 CQ: load CQ MMIO 地址，检查 CQE

GPU1 侧:
  - NIC 收到数据 → RDMA WRITE 直接写入 GPU1 HBM 的对称窗口
  - GPU1 完全被动，CPU 不用参与
  - GPU1 Kernel 可以轮询 flag/计数器确认数据到达
```

### 6.3 混合拓扑 (LSA + GIN)

对于 4 节点 × 8 GPU 集群，做 All-to-All：

```text
Node0:  [GPU0] [GPU1] ... [GPU7]
           │ LSA              │ GIN (via NIC)
Node1:  ...              [GPU15]
           │                  │
Node2:  ...              [GPU23]
           │                  │
Node3:  ...              [GPU31]

GPU0 的通信策略:
  → GPU0-7 (同节点):  LSA (NVLink, ~几百 ns 延迟)
  → GPU8-31 (跨节点): GIN (RDMA,  ~μs 延迟)

总 QP 数: 与之前 Ring 分析一致，每卡 ~O(1) QP
```

---

## 7. 硬件要求

| 组件 | 最低要求 | 说明 |
| ---- | -------- | ---- |
| GPU | **NVIDIA H100 (Hopper)** | SM90+，支持 BAR1 映射网卡 MMIO |
| 网卡 | **ConnectX-7** 及以上（或 BlueField-3） | 支持 MMIO doorbell 映射到 GPU 地址空间 |
| 互联 | PCIe Gen5 / NVLink-C2C | GPU-NIC 间带宽足够支撑 MMIO 访问 |
| CUDA | 12.0+ | GDAKI API 支持 |
| NCCL | 2.18+ | 官方 GIN 支持 |

**为什么需要 Hopper？** 之前的 GPU（Ampere/A100）虽然支持 GPUDirect RDMA（NIC 可以 DMA 读写 GPU 显存），但 GPU 不能主动发 MMIO 请求给 NIC。Hopper 的 NVLink-C2C 和增强的 BAR1 映射能力使 GPU 可以直接访问 NIC 的控制寄存器。

---

## 8. 性能收益

| 指标 | 传统 NCCL (CPU Proxy) | GIN | 提升 |
| ---- | --------------------- | --- | ---- |
| **Small I/O 延迟** | ~5-10 μs | ~2-5 μs | 40-50% |
| **CPU 占用** | 每 QP 一个轮询线程 | 0 | 100% |
| **Kernel 融合** | 不支持 | 支持 | 消除 kernel launch 开销 |
| **All-to-All 带宽** (跨节点) | 受 CPU 转速限制 | 接近线速 | ~10-20% |

GIN 最大的收益场景：

1. **Small I/O 密集**：每跳数据小但跳数多，CPU 开销占比大 → GIN 消除 CPU 开销
2. **Compute + Send 融合**：GPU 算完直接发，不用等 CPU 调度 post send
3. **大规模 All-to-All**：千卡场景，CPU 轮询线程数以千计 → GIN 全部省掉

---

## 9. 限制与局限

- **Barrier 同步**：GIN 只是发送/接收通知，如需全局 barrier 仍需 CPU 侧参与或 GPU 侧额外同步
- **错误处理**：网卡出错时 GPU Kernel 难以处理（不像 CPU 有完整的错误恢复路径），通常仍需 CPU 兜底
- **调试困难**：GPU Kernel 直接写 NIC 门铃，cuda-gdb 难以调试 doorbell 到数据 DMA 的完整链路
- **RC QP 限制**：GIN 目前主要使用 RC QP，无法像 UD 那样一个 QP 发向多个目标（需为每个 peer 建 QP）
- **Memory Registration**：仍需 CPU 预先注册 MR，GIN 只省了 post send 这一步，不能跳过初始化

---

## 10. 总结

```text
┌───────────────────────────────────────────────────────┐
│                    NCCL 通信层次                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│  高层 API:  ncclAllReduce / ncclAlltoAll / ...        │
│                  ▲                                     │
│  算法层:     Ring / Tree / CollNet                    │
│                  ▲                                     │
│  传输层:     ┌───────────┬───────────┐                │
│              │   LSA     │    GIN    │                │
│              │ (节点内)   │ (跨节点)  │                │
│              │ NVLink    │ RDMA 网卡 │                │
│              │ load/store│ doorbell  │                │
│              └───────────┴───────────┘                │
│                  ▲                                     │
│  硬件层:     NVSwitch  │  InfiniBand / RoCE           │
│                                                       │
└───────────────────────────────────────────────────────┘

一句话: GIN 让 GPU 可以直接按网卡门铃，LSA 让 GPU 可以直接摸远端显存。
       两者组合，Hopper 时代的分布式通信不再需要 CPU 代理。
```
