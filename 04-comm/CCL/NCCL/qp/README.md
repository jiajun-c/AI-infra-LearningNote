# QP

## 1. 概念

- SQ(Send Queue): 存放你要发送的数据的指令
- RQ(Receive Queue): 存放准备接收数据的内存缓冲区

QP通常是成对出现的，机器A的一个QP需要和机器B的一个QP建立连接(常见的是RC，类似TCP的可靠按序传输)

## 2. QP 类型（Transport Types）

RDMA 定义了四种传输类型，按"可靠/不可靠"和"面向连接/无连接"两个维度划分：

| 类型 | 全称 | 可靠 | 有序 | 面向连接 | 类比 | NCCL 是否使用 |
| ---- | ---- | ---- | ---- | -------- | ---- | ------------ |
| **RC** | Reliable Connection | ✅ | ✅ | ✅ | TCP | ✅ 主要使用 |
| **UC** | Unreliable Connection | ❌ | ✅ | ✅ | — | ❌ |
| **RD** | Reliable Datagram | ✅ | ✅ | ❌ | — | ❌ |
| **UD** | Unreliable Datagram | ❌ | ❌ | ❌ | UDP | ✅ 部分场景 |

- **RC**：最常用，保证可靠按序交付，需要连接建立/拆毁，NCCL 的 Ring 和 Tree 算法都用 RC
- **UD**：无连接，单包大小受限（MTU），NCCL 的 bootstrap 阶段和 allreduce 某些实现使用

## 3. QP 状态机

QP 不是创建即可用的，必须经过严格的状态迁移。每个状态的进入都通过 `ibv_modify_qp()` 完成：

```
                          ┌──────────┐
                          │  RESET   │  ← 初始/重置状态
                          └────┬─────┘
                               │ ibv_modify_qp(INIT)
                               ▼
                          ┌──────────┐
                          │   INIT   │  ← 指定 PD、CQ、端口
                          └────┬─────┘
                               │ ibv_modify_qp(RTR) + 对端信息
                               ▼
                          ┌──────────┐
                          │   RTR    │  ← Ready to Receive，可以接收
                          └────┬─────┘
                               │ ibv_modify_qp(RTS)
                               ▼
                          ┌──────────┐
                          │   RTS    │  ← Ready to Send，可以收发
                          └────┬─────┘
                               │ 错误发生
                               ▼
                   ┌─────────────────────┐
                   │  SQ Error / Error   │  ← 错误状态
                   └─────────────────────┘
```

关键迁移条件：
- **INIT → RTR**：需要指定对端的 QP 编号(QPN)、LID、端口号等连接信息
- **RTR → RTS**：需要指定超时、重试次数等参数

---

## 4. QP 属性（Capability）

创建 QP 时需要指定以下关键参数，决定了 QP 的容量和行为：

| 属性 | 含义 | 说明 |
| ---- | ---- | ---- |
| **max_send_wr** | SQ 最大深度 | SQ 环形缓冲区最多容纳多少个 WQE，决定了最大 in-flight 发送数 |
| **max_recv_wr** | RQ 最大深度 | RQ 环形缓冲区最多容纳多少个 WQE |
| **max_send_sge** | 每 WQE 最大 SGE 数 | 一个发送 WQE 可以描述多少个不连续内存片段（scatter） |
| **max_recv_sge** | 每 WQE 最大 SGE 数 | 一个接收 WQE 可以描述多少个不连续内存片段（gather） |
| **max_inline_data** | 内联发送上限 | 小数据量时把 payload 直接嵌入 WQE，省一次 DMA 读内存 |

---

## 5. 指令载体（WQE 与 WR）

- WR(Work Request，工作请求)：在软件层面，当你想要发送或者接收数据时，会构建一个 WR，描述了通信的关键要素，包括从哪个内存取，取多长的数据，发送给谁
- WQE(Work Queue Element)：当你的 WR 提交(Post)给网卡(HCA, Host Channel Adapter)后，网卡硬件内部将其称为 WQE，WQE 也就是实际存在于 SQ 或者 RQ 环形缓冲区的数据结构

---

## 6. Work Request 的四种主要操作

| WR 类型 | 方向 | 需要远端 CPU 参与 | 需要远端 rkey | 说明 |
| -------- | ---- | ------------------ | -------------- | ---- |
| **SEND** | 发送 | 需要对端预先 post RECV | ❌ | 类似 TCP send，双边操作 |
| **RECV** | 接收 | 被动接收 | ❌ | 预投递空缓冲区，等待对端 SEND 填入 |
| **RDMA WRITE** | 发送 | ❌ 不需要 | ✅ | 直接写远端已注册内存（需要目标地址 + rkey） |
| **RDMA READ** | 接收 | ❌ 不需要 | ✅ | 直接从远端已注册内存读取（需要源地址 + rkey） |
| **ATOMIC** | 远端 | ❌ 不需要 | ✅ | CAS(Compare & Swap)、Fetch & Add |

**双边 vs 单边**：
- 双边（SEND/RECV）：对端 CPU 必须预先 post RECV 到 RQ，否则 SEND 失败（RNR retry）。适合流式、控制消息
- 单边（RDMA WRITE/READ）：完全绕过远端 CPU，只需知道远端内存的 rkey 和虚拟地址。适合大数据块传输，NCCL 大量使用

---

## 7. Scatter-Gather Element (SGE)

每个 WQE 内部包含一个 SGE 数组，描述实际的内存位置：

```
struct ibv_sge {
    uint64_t addr;   // 内存虚拟地址（必须是已注册 MR 内的地址）
    uint32_t length; // 该片段长度
    uint32_t lkey;   // 本地 MR 的 lkey，证明访问权限
};

struct ibv_send_wr {
    ...
    struct ibv_sge *sg_list;  // SGE 数组指针
    int             num_sge;  // SGE 个数
    ...
};
```

SGE 让一个 WQE 可以从不连续的多块内存聚合发送（gather），或分散接收（scatter），避免 CPU 手动拷贝拼接。

---

## 8. Memory Region (MR)

RDMA 网卡只能访问**已注册**的内存，这是 RDMA 的核心安全机制：

```
ibv_reg_mr(pd, addr, length, access_flags)
    → 获得 mr（含 lkey 和 rkey）
```

- **注册**：用户态调用 `ibv_reg_mr()` 将虚拟内存地址区间告知 HCA，HCA pin 住物理页并建立虚拟→物理映射表
- **lkey**：本地 key，WR 的 SGE 必须携带，证明本地内存的访问权限
- **rkey**：远端 key，做 RDMA WRITE/READ 时必须提供对端的 rkey，证明远端允许你访问
- **access_flags**：控制权限（`IBV_ACCESS_LOCAL_WRITE`、`IBV_ACCESS_REMOTE_WRITE`、`IBV_ACCESS_REMOTE_READ`、`IBV_ACCESS_REMOTE_ATOMIC`）

⚠️ **与 NCCL 的关系**：NCCL 在初始化阶段通过 bootstrap 通道交换各 rank 的内存注册信息（地址 + rkey），之后 GPUDirect RDMA 才能直接访问远端 GPU 显存。

---

## 9. 异步状态通知

RDMA 是完全异步的，CPU 将 WQE 扔到 QP 后就可以立即返回去做别的工作。

- **CQ**（Completion Queue）：完成队列，一个或多个 QP 的 SQ 和 RQ 可以绑定到同一个 CQ 上
- **CQE**（Completion Queue Entry）：当一个 WQE 成功或失败处理完毕，网卡硬件会主动向对应的 CQ 中写入一个 CQE

### Polling 模式

NCCL 不使用中断，而是用 **Polling 模式** 不断检查 CQ：

```c
// 轮询 CQ，检查是否有新的完成事件
struct ibv_wc wc;
int ret = ibv_poll_cq(cq, 1, &wc);

if (ret > 0) {
    if (wc.status == IBV_WC_SUCCESS) {
        // 操作成功，wc.wr_id 可定位是哪个 WR 完成了
    } else {
        // 操作失败，wc.status 包含错误码
    }
}
// ret == 0 表示暂无完成事件，CPU 继续做其他事或自旋
```

中断 vs Polling 的选择：

| 方式 | 延迟 | CPU 开销 | 适用场景 |
| ---- | ---- | -------- | -------- |
| **中断** | 较高（μs 级） | 低 | 通用、低频通信 |
| **Polling** | 极低（ns 级） | 高（100% 自旋） | HPC、NCCL、低延迟场景 |

---

## 10. NCCL 中的 QP 使用

放在 NCCL 目录下的补充——QP 在 NCCL 中如何使用：

### QP 数量

- NCCL 通常为**每对 GPU 之间创建多个 QP**（例如 16-32 个），增加并行度，充分利用多队列硬件能力
- 每个 QP 可以绑定到独立的 CQ，也可以多个 QP 共享一个 CQ

### QP 数量是否会爆炸？——千卡 A2A 场景分析

**直觉陷阱**：1000 张卡做 All-to-All，如果按全连接 (full mesh) 建 QP，情况是：

```
全连接 QP 数（理论）:
  总 QP 对数 = C(1000, 2) = 1000 × 999 / 2 ≈ 499,500 对
  每对 16 个 QP → 总 QP ≈ 799 万个
  每张卡: 999 个 peer × 16 QP ≈ 16,000 QP/GPU
```

这确实会爆炸。每张卡维护 16000 个 QP，光是 QPC (QP Context) 就能耗尽 HCA 的片上 SRAM（ConnectX-7 约支持几万 QP，但不是无限）。

**实际情况——NCCL 并不会全连接建 QP**：

#### 方案一：Ring/Tree 拓扑 —— QP 数量 O(1)

NCCL 的 Ring AllReduce 和 Ring 类算法中，每张卡只和环中相邻的 **2 个 peer** 通信：

```
千卡 Ring 的 QP 数:
  每卡: 2 peers × 16 QP/peer = 32 QP（双向各 16）
  总计: 1000 × 32 = 32,000 QP（线性增长，非平方）
```

Tree 算法同理，每张卡只与 parent + children（共 O(log N) 个邻居）建立 QP。

**结论**：Ring/Tree 拓扑下 QP 数量线性增长 O(N)，不会爆炸。

#### 方案二：真正的 All-to-All (如 MoE 的 expert dispatch)

当确实需要每张卡向其他 999 张卡发数据时，有几种策略避免 QP 爆炸：

| 策略 | 原理 | QP 数量 | 代价 |
| ---- | ---- | ------- | ---- |
| **分段 Ring A2A** | 将 A2A 拆成多个 Ring 阶段，数据通过环逐跳转发 | 每卡 O(1) QP | 延迟增加（多跳） |
| **对等合并** | 每张卡只与 K 个物理 peer 建 QP（K << N），远端的流量经中间节点转发 | 每卡 O(K) QP | 需要转发，增加带宽压力 |
| **分层通信** | 节点内用 NVLink/NVSwitch（无 QP），跨节点才用 RDMA | 跨节点 QP 数大幅减少 | 架构耦合 |
| **动态连接 (DCT)** | NCCL 2.18+ 引入，QP 按需创建/销毁，不维持全时连接 | 活跃 QP 数可控 | 建立/拆除有开销 |

#### 方案三：利用 RDMA WRITE（单边）

All-to-All 场景中，如果所有 rank 预先交换了内存注册信息（地址 + rkey），可以直接用 RDMA WRITE 单边写入：

```
RDMA WRITE 不需要对端 post RECV:
  每卡可以 RDMA WRITE 到任意远端（有 rkey 即可）
  不需要为每个目标预建 QP（可以复用同一个 QP 的 SQ 发向不同目标）
```

但注意：RC QP 是面向连接的，一个 QP 只能连接一个对端 QP。所以要发向不同远端仍然需要不同的 QP。不过 **UD (Unreliable Datagram)** QP 可以发送到任意对端，适合 A2A 这种场景（代价是失去了可靠性保证，需要上层处理丢包）。

#### 千卡 A2A 实际数字估算

假设 8 机 × 8 卡架构（64 GPU/节点）：

| 层级 | 通信方式 | QP 情况 |
| ---- | -------- | ------- |
| **节点内** (8 GPU) | NVLink/NVSwitch | 0 个 RDMA QP，走 NVLink |
| **跨节点** (16 节点) | RDMA (RoCE/IB) | 每卡只需连接其他 15 个节点的对等卡 |

如果采用 Ring 跨节点（每节点只和 prev/next 节点通信）：

```
跨节点 Ring:
  每卡 : 2 个 peer × 16 QP = 32 QP（跨节点部分）
  加上节点内: 0 QP（NVLink）
  总计/卡: ~32 QP  ← 完全可控
```

如果要求真正的全对全、每张卡直接访问每张远端卡（不经过环转发）：

```
直接连接:
  每卡 : (16-1) 个远端节点 × 1 对等卡 = 15 peers
  每卡 : 15 peers × 16 QP = 240 QP
  总计 : 1000 × 240 = 240,000 QP
```

240 QP/卡 仍然在 ConnectX-7 的承受范围内（通常支持几千到上万 QP），但关键是 — **实际部署中几乎不会这样设计**，Ring/转发已经把 QP 数压到 O(1) 了。

#### 核心结论

> **千卡 A2A 场景下 QP 不会爆炸。** NCCL 通过 Ring/Tree 拓扑让每卡 QP 数保持 O(1)（与集群规模无关），而非 O(N) 全连接。即便需要真正的全对全通信，分层架构（节点内 NVLink + 跨节点 RDMA）和动态连接技术也把 QP 数压在实际可承受范围内。

### 传输语义选择

| NCCL 算法 | 主要使用 | 说明 |
| --------- | -------- | ---- |
| **Ring** | SEND/RECV（双边） | 数据在环中流式传递，对端需预先 post RECV |
| **Tree** | SEND/RECV（双边） | 树形归约/广播 |
| **某些实现** | RDMA WRITE（单边） | 直接写入远端 GPU 显存，绕过远端 CPU，降低延迟 |

### 完整数据流

```
GPU 显存
    ↓ GPUDirect RDMA（绕过 CPU，直接 PCIe/NVLink 到网卡）
HCA 网卡
    ↓ SQ WQE 描述的数据
InfiniBand / RoCE 网络
    ↓
远端 HCA 网卡
    ↓ RQ WQE 描述的目标缓冲区 或 直接 RDMA WRITE
远端 GPU 显存
```

### 连接建立流程（简化）

```
1. NCCL Bootstrap（TCP/共享内存）交换初始信息
2. 各 rank 注册 GPU 显存 MR（ibv_reg_mr）
3. 通过 Bootstrap 交换 MR 信息（地址 + rkey）
4. 创建 QP，状态迁移：RESET → INIT → RTR → RTS
5. 对端信息（QPN、LID）通过 Bootstrap 交换
6. QP 进入 RTS 状态，开始通信
```

---

## 11. 整体流程图

```
┌─────────────────────────────────────────────────────────┐
│ 应用层                                                   │
│  构建 WR → ibv_post_send(post_recv) → 立即返回           │
│    │                                                     │
│    ▼                                                     │
│ ┌──────────┐      ┌──────────┐                          │
│ │    SQ    │      │    RQ    │   QP（Queue Pair）         │
│ │  WQE[]   │      │  WQE[]   │                          │
│ └────┬─────┘      └────┬─────┘                          │
│      │                 │                                 │
│      ▼                 ▼                                 │
│ ┌─────────────────────────────┐                          │
│ │         HCA（网卡）          │                          │
│ │  处理 WQE，DMA 读写内存      │                          │
│ └──────────┬──────────────────┘                          │
│            │ 完成时写入                                    │
│            ▼                                             │
│ ┌──────────────────┐                                     │
│ │       CQ         │   完成队列                            │
│ │      CQE[]       │                                     │
│ └──────┬───────────┘                                     │
│        │                                                 │
│        ▼                                                 │
│  应用 Polling CQ 取出 CQE                                 │
│  （ibv_poll_cq）确认完成                                  │
└─────────────────────────────────────────────────────────┘
```
