# 全局内存合并访问

全局内存（HBM/GDDR）是 GPU 上容量最大但延迟最高的内存。性能关键不在于避开某条指令，而在于**让每次内存事务的效率最大化**。

## 1. 内存事务模型

GPU 全局内存不是按字节读写的，而是按 **transaction**（事务）：

```
Transaction 粒度:
  L1 命中:     32 字节
  L2 cache line: 128 字节 (最常见)
```

一次 transaction 总是传输完整的 128 字节。如果你的 warp 只用到了其中 4 字节，剩余 124 字节就浪费了。

## 2. 地址合并过程

Warp 发出 32 个地址后，LSU 的合并逻辑：

```text
           Warp 32 个地址
                │
                ▼
    ┌──────────────────────────┐
    │   LSU 地址合并单元        │
    │  1. 找到最小/最大地址      │
    │  2. 按 128B 对齐分段       │
    │  3. 每段 → 1 transaction  │
    └──────────────────────────┘
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
     tx #0    tx #1    tx #N
```

**核心**：不是主动把地址合并到一起，而是用**最少的 transaction 数覆盖所有地址**。

## 3. Stride 对 Transaction 数的影响

以 32 个线程、每个读 4B（float）为例：

```
stride=1: 32 地址连续 → 覆盖 128B → 1 transaction
stride=2: 32 地址跨 256B → 2 transactions
stride=4: 32 地址跨 512B → 4 transactions
stride=8: 32 地址跨 1024B → 8 transactions
stride=32: 每个地址在不同 128B 段 → 32 transactions
```

**公式**（stride ≤ 32 时）：

```text
transactions = stride
利用率 = 32 × 4B / (stride × 128B) = 1 / stride
```

### 以 stride=8 为例，逐线程分解

```text
T0 → offset 0    ─┐
T1 → offset 32    ├─ [0,   127]  → transaction 1
T2 → offset 64    │  128B 中用到 4×4=16B (12.5%)
T3 → offset 96   ─┘
─────────────────────────
T4 → offset 128  ─┐
T5 → offset 160   ├─ [128, 255]  → transaction 2
T6 → offset 192   │  同上, 16B/128B
T7 → offset 224  ─┘
─────────────────────────
... 依此类推, 共 8 个 transaction
```

## 4. Random Access：四重惩罚

最坏情况 = 随机地址访问。不仅 transaction 层面浪费，还触发缓存和 DRAM 层面的连锁惩罚：

| 层次 | 问题 | 额外延迟 |
|------|------|---------|
| **Transaction** | 32 线程 × 128B = 4KB 传输, 只用 128B → 3.1% 利用率 | ~2-5× |
| **L2 Cache** | 无时间/空间局域性 → 全部 miss, 每次走 DRAM | ~200-800 cycles |
| **TLB** | 跨大量内存页 → TLB miss → 页表遍历 | ~1-2 次额外访存 |
| **DRAM Row Buffer** | 每次访问不同 row → activate + precharge 循环 | DRAM 带宽再降 50-80% |

### 为什么 stride=32 比 random 快

虽然 stride=32 也需要 32 transactions，但地址是线性递增的，DRAM 层面仍有局域性：

```text
stride=32:  tx#0→addr0, tx#1→addr1024, tx#2→addr2048 ...
            地址规律 → DRAM row buffer 可能命中

random:     tx#0→0xFFA3, tx#1→0x0012, tx#2→0xABCD ...
            每次跳随机 row → row buffer 全部 miss
```

## 5. SoA vs AoS

这部分访问模式直接影响 LLM kernel 设计：

```text
SoA (Structure of Arrays):          AoS (Array of Structures):
  x = [x0,x1,x2,...]                  v = [{x0,y0,z0}, {x1,y1,z1}, ...]
  y = [y0,y1,y2,...]
  z = [z0,z1,z2,...]

  读 x → 连续访问, 1 transaction       读 v.x → stride=3, 3 transactions
```

**LLM 中的实践**：KV cache 按 head 维度分开存（SoA），不按 token 的完整 KV struct（AoS），原因就在这里。

## 6. 与共享内存 Bank Conflict 的区别

| | 共享内存 Bank Conflict | 全局内存不合并 |
|---|---|---|
| **粒度** | 4 字节/线程 | 128 字节/transaction |
| **触发条件** | 同 bank 不同地址 | 地址跨多个 128B 段 |
| **惩罚** | warp 内串行化 (2-32 cycle) | 多 transaction + cache miss (>1000 cycle) |
| **优化** | pad + 转置 | 连续访问 + SoA 布局 |

关键差异：全局内存的不合并比共享内存的 bank conflict 代价**大得多**——不仅多花 transaction，还会逐级穿透 L1 → L2 → DRAM → row buffer。

## 参考

- [memhit.cu](./memhit.cu)：实测 benchmark，对比 stride=1~32、random access 的带宽
- [CUDA Programming Guide §5.3 - Maximize Memory Throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput)

## 7. 实战：GEMV 行主序 vs 列主序

GEMV（矩阵乘向量）`y = A @ x` 是 LLM 推理中最常见的算子之一（Attention 的 projection、MLP 的 FC 层在 decode 阶段 batch=1 时都是 GEMV）。行主序和列主序的选择直接决定 coalescing 表现。

### 存储布局

```text
矩阵 A 是 M 行 × N 列

行主序 (Row-major):                 列主序 (Column-major):
  内存: [A₀₀ A₀₁ ... A₀ₙ₋₁ |         内存: [A₀₀ A₁₀ ... Aₘ₋₁₀ |
         A₁₀ A₁₁ ... A₁ₙ₋₁ |                A₀₁ A₁₁ ... Aₘ₋₁₁ |
         ...             |                ...             |
         Aₘ₋₁₀ Aₘ₋₁₁ ... Aₘ₋₁ₙ₋₁]        A₀ₙ₋₁ A₁ₙ₋₁ ... Aₘ₋₁ₙ₋₁]

  相邻线程读同一行的相邻元素 → 合并    相邻线程读不同行的同列元素 → 不合并
```

### Warp 视角：一个 warp 的 32 个线程同时读 A

**行主序 + 按行并行（每个线程负责矩阵的一行）：**

```text
T0 负责 row 0: 读 A[0][0], A[0][1], A[0][2] ...
T1 负责 row 1: 读 A[1][0], A[1][1], A[1][2] ...

但 T0 和 T1 同时执行第一条 LDG:
  T0 → A[0][0] = offset 0          ─┐
  T1 → A[1][0] = offset N          ─┼─ 间隔 N 个 float = 4N 字节
  T2 → A[2][0] = offset 2N          │
  ...                                │ stride = N
  T31→ A[31][0] = offset 31N       ─┘

  如果 N ≥ 32: 每个地址在不同 128B 段 → 32 transactions → 最差!
```

这就是**按行并行的经典坑**：即使矩阵是行主序存储，但只要 warp 内 32 个线程各负责一行，它们同时读 A 的第一列，stride 就是 N（矩阵宽度），如果 N ≥ 32 就是最差不合并。

**列主序 + 按列并行（每个线程负责矩阵的一列）：**

```text
矩阵以列主序存储 (FORTRAN 风格, cuBLAS 默认)

T0 负责 col 0: 读 A[0][0], A[1][0], A[2][0] ...
T1 负责 col 1: 读 A[0][1], A[1][1], A[2][1] ...

T0 和 T1 同时执行第一条 LDG:
  T0 → A[0][0] = offset 0          ─┐
  T1 → A[0][1] = offset M          ─┼─ stride = M (行数)
  T2 → A[0][2] = offset 2M          │
  ...                                │ stride = M
  T31→ A[0][31] = offset 31M       ─┘

  同样 stride = M, 如果 M ≥ 32 → 不合并 → 最差!
```

### 真正的解决方案

两种经典策略，对应 LLM 推理中 GEMV 的实际实现：

**策略 1: 改变 warp 调度——让连续线程读相邻元素而非相邻行**

```text
不要: T0→row0, T1→row1, T2→row2 ...
应该: T0→row0_col0, T1→row0_col1, T2→row0_col2 ...

  行主序下:
  T0→A[0][0], T1→A[0][1], T2→A[0][2] ... T31→A[0][31]
  → 全部连续, 1 transaction!
```

这就是**向量化加载**的基础——32 个线程合作加载一行，每 32 列一段。LLM 中 `hidden_dim=4096` 时，每行 4096 个 float 被 128 个线程（4 个 warp）合作处理，每个 warp 内 32 个线程连续读一行中相邻的 32 个元素 → 完美合并。

**策略 2: 转置存储——存储布局与访问方向匹配**

```text
如果访问总是按列, 就让矩阵以列主序存储
如果访问总是按行, 就让矩阵以行主序存储

LLM 推理: 权重矩阵 W 存储为 V^T (转置)
  W: [in_dim, out_dim], 行主序
  实际读取 W^T: [out_dim, in_dim], 行主序
  → 每行加载 out_dim 个 float, warp 内连续 → coalesced
```

这就是 PyTorch 里 `nn.Linear` 将 weight 存为 `[out_features, in_features]` 的原因。

### 可视化对比

```text
行主序矩阵, warp 按行并行 (stride=N):
┌───────────────────────────────────┐
│ T0 → ◆ □ □ □ □ □ □ □ □ □ □ □ □  │  每个 ◆ 间隔 N 列
│ T1 → ◇ □ □ □ □ □ □ □ □ □ □ □ □  │  64 字节 (假设 N=16 个 float)
│ T2 → ○ □ □ □ □ □ □ □ □ □ □ □ □  │  → 不合并 ❌
│ ...                              │
└───────────────────────────────────┘

行主序矩阵, warp 按列分段并行 (stride=1):
┌───────────────────────────────────┐
│ T0→◆ T1→◇ T2→○ T3→△ ... T31→□ │  全部连续
│ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □  │  → 1 transaction ✅
│ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □  │
│ ...                              │
└───────────────────────────────────┘
```

### 核心结论

| 情况 | 每个线程读 | warp 内 stride | transactions |
| ---- | ---------- | -------------- | ------------ |
| 行主序 + 线程按行 + 大 N | `A[i][0]`, `A[i][1]`, ... | N | ~32 (最差) |
| 行主序 + 线程合作一行 | `A[0][0..31]`, `A[0][32..63]`, ... | 1 | 1 (最好) |
| 列主序 + 线程按列 + 大 M | `A[0][j]`, `A[1][j]`, ... | M | ~32 (最差) |

**关键是 warp 内线程的地址分布，不是矩阵的存储布局。** 存储布局是你选的，warp 线程映射也是你选的——两者要匹配。
