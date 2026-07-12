# EP并行

Expert Parallelism（EP）是 MoE 架构的专属并行策略。核心思想：**将不同的 Expert 放置在不同的 GPU 上，每个 GPU 只计算分配给它的 Expert，token 通过通信路由到对应专家所在的 GPU**。

EP 不同于 DP/TP/PP——它切的是 **Expert**，通信的是 **token 的 hidden state**（而非梯度或 activation）。

## 问题的本质

MoE 层中，每个 token 只激活 K 个专家（如 K=2）。当专家总数 N=8、有 4 张 GPU 时，每张 GPU 放置 2 个专家：

```text
┌─────────────────────────────────────────────────────┐
│  GPU 0: Expert 0, Expert 1                          │
│  GPU 1: Expert 2, Expert 3                          │
│  GPU 2: Expert 4, Expert 5                          │
│  GPU 3: Expert 6, Expert 7                          │
└─────────────────────────────────────────────────────┘
```

问题在于：GPU 0 上的 token 可能被路由到 Expert 5（在 GPU 2 上）。因此需要把 token 发到 GPU 2 去计算，再把结果送回来。

## EP 三个阶段

EP 一次 MoE 层的完整计算分为三个步骤：

```text
      dispatch          compute           combine
    ───────────→     ───────────→     ───────────→
    token 按路由       各 GPU 对本地       将计算结果
    发送到目标 GPU      Expert 计算        送回原 GPU
```

### 第一阶段：Dispatch（分发）

Router 计算出门控分数后，每个 GPU 确定自己持有的 token 需要被发往哪些 GPU。

```text
Dispatch 示意 (4 GPU, 8 Expert, TopK=2):

   GPU 0 持有的 tokens:
   ┌────────────────────────────────────────┐
   │ t0 → Expert 2,5  │ 发往 GPU1, GPU2     │
   │ t1 → Expert 0,7  │ 发往 GPU0, GPU3     │
   │ t2 → Expert 1,4  │ 发往 GPU0, GPU2     │
   │ t3 → Expert 6,3  │ 发往 GPU3, GPU1     │
   └────────────────────────────────────────┘

   GPU 0 发送:
   ┌─────────────────┐
   │ GPU0: t1(x), t2(x)    (本地 Expert 0,1) │
   │ GPU1: t0(x), t3(x)    (Expert 2,3)     │
   │ GPU2: t0(x), t2(x)    (Expert 4,5)     │
   │ GPU3: t1(x), t3(x)    (Expert 6,7)     │
   └─────────────────┘
```

关键细节：

- 每个 token 被复制 K=2 份，分别发往不同 GPU
- 同一个 GPU 上的多个 token 按目标 GPU 分组打包
- 通信模式是 **All-to-All**（每个 GPU 都向其他 GPU 发送不同数据）

### 第二阶段：Compute（计算）

每个 GPU 收到来自所有 GPU 的 token 后，对本地的 Expert 执行计算：

```text
GPU 0 收到的 tokens            本地 Expert 计算
┌──────────────────┐     ┌─────────────────────┐
│ t1 (来自 GPU0)   │────→│ Expert 0(t1) → y1₀  │
│ t2 (来自 GPU0)   │────→│ Expert 1(t2) → y2₁  │
│ t5 (来自 GPU1)   │──┐  │                     │
│ t7 (来自 GPU2)   │──┤  │ Expert 0(t5) → y5₀  │
│ t9 (来自 GPU3)   │──┘  │ Expert 0(t9) → y9₀  │
│ ...              │     │ ...                 │
└──────────────────┘     └─────────────────────┘
```

关键细节：

- 每个 GPU 只计算本地的几个 Expert（如 2/8），不碰其他 Expert 的参数
- 这是 EP 节省计算量的核心：每 token 的实际计算量 = K × (单个 Expert FLOPs)，与总 Expert 数 N 无关
- 不同 Expert 的输入可以 batch 在一起并行计算

### 第三阶段：Combine（聚合）

计算结果通过 **All-to-All** 送回 token 原 GPU，然后加权求和：

```text
Combine 示意:

GPU 0 持有的结果                         最终输出
┌──────────────────────┐         ┌─────────────────────┐
│ y1₀ (Expert 0 产出)  │         │                     │
│ y2₁ (Expert 1 产出)  │         │ out(t1) = w₀·y1₀   │
│ y5₀ (Expert 0 产出) ─│─送回──→ │         + w₇·y1₇   │
│ y9₀ (Expert 0 产出)  │         │                     │
│ y1₇ (来自GPU3)       │         │ out(t2) = w₁·y2₁   │
│ y2₄ (来自GPU2)       │         │         + w₄·y2₄   │
│ ...                  │         │ ...                 │
└──────────────────────┘         └─────────────────────┘

out(token) = Σ g_i · E_i(token)   (对 TopK 个专家加权求和)
```

关键细节：

- 每个 token 的 K 个专家输出从不同 GPU 返回
- 聚合时用 Router 的 gating score 做加权
- residual connection（残差连接）在 EP 层外完成，保证即使 token 溢出也有一条直通路径

## 通信核心：All-to-All

EP 的 dispatch 和 combine 阶段都使用 **All-to-All** 通信：

```text
All-to-All 在 EP 中的作用：

    GPU0    GPU1    GPU2    GPU3
    ┌─┐     ┌─┐     ┌─┐     ┌─┐
    │0│     │1│     │2│     │3│
    └┬┘     └┬┘     └┬┘     └┬┘
     ├──────→┼──────→┼──────→┼──→ 按 Expert 分发 token (dispatch)
     │       │       │       │
     ├←──────┼←──────┼←──────┼──→ 按 Token 送回结果 (combine)
     │       │       │       │
    └─┘     └─┘     └─┘     └─┘

从矩阵视角看，All-to-All = 矩阵转置：
  Dispatch: [batch, d] 按行分块 → A2A → 按 expert 分块
  Combine:  [expert, d] 按行分块 → A2A → 按 batch 分块
```

详细原理见 [分布式转置](../trans/README.md)。

## 专家放置策略

### 均匀放置

每个 GPU 上放置相同数量的专家：

```text
GPU数=P, 专家数=N, 每GPU放置 N/P 个专家（N 需被 P 整除）

例如 N=8, P=4:
GPU0: E0 E1    GPU1: E2 E3
GPU2: E4 E5    GPU3: E6 E7
```

优点：负载天然均匀（假设 Router 也均匀）。
缺点：需要 N 是 P 的倍数，扩展性受限。

### 非均匀放置

当 N 不能被 P 整除，或出于负载均衡考虑：

```text
N=10, P=4:
GPU0: E0 E1 E2    (3个)
GPU1: E3 E4 E5    (3个)
GPU2: E6 E7       (2个)
GPU3: E8 E9       (2个)
```

注意：专家负载不同时，需要更复杂的负载均衡策略。

## 负载均衡

EP 面临的核心挑战是 **token 分布不均**——如果 Router 把所有 token 都发给少数几个专家，部分 GPU 会过载，其他 GPU 空闲。

### 辅助损失（Auxiliary Loss）

```text
L_aux = N · Σ f_i · P_i

其中：
- f_i: 第 i 个专家实际处理的 token 比例
- P_i: Router 分配给第 i 个专家的平均概率

最优时 f_i = P_i = 1/N，此时 L_aux 最小
```

### 容量因子（Capacity Factor, CF）

为每个专家设置处理上限，超出部分直接丢弃（走 residual）：

```text
Expert Capacity = (total_tokens / N) × CF

当 CF = 1.0: 每个专家处理约 batch_size/N 个 token
当 CF = 1.25: 允许 25% 的溢出余量（常见设置）
当 CF = ∞: 不做限制（实验阶段使用）
```

### Token 丢弃（Token Dropping）策略

当 token 超过某专家的 capacity 时：

- **Switch Transformer**: 直接丢弃（让 residual 连接兜底）
- **GShard**: 尝试发给次选专家
- **DeepSeek-V3**: 使用无辅助损失的动态偏置路由

## EP 与其他并行策略的组合

实际大模型训练中，EP 通常与其他并行策略联用：

```text
                    TP (层内)
                    ┌──────┐
                    │G0 G1 │ ← 同一层内 TP 切分
                ┌───┴──────┤
    EP          │ Expert 0,1│
    ┌──────┐    │           │
    │G0 G1 │    │ Expert 2,3│
    │  E0  │    └──────────┘
    │  E1  │
    │G2 G3 │    PP (层间)
    │  E2  │    ┌──────────┐
    │  E3  │    │ Layer 1  │
    └──────┘    ├──────────┤
                │ Layer 2  │
                └──────────┘
           DP (数据并行)
    ┌──────┐  ┌──────┐
    │ DP-0 │  │ DP-1 │  ... 不同 dp group 处理不同 batch
    └──────┘  └──────┘
```

常见组合：

- **EP + DP**: 每个 EP group 处理一个 batch 分片，不同 EP group 间做 DP
- **EP + TP**: 专家内部有较大的 FFN，用 TP 进一步切分
- **EP + PP**: 不同层的专家分布在不同设备上

## 实际案例分析

### Mixtral 8×7B

```text
配置: 8 专家, TopK=2
每 token 激活: 2/8 = 25% 的 FFN 参数
总参数 ~47B, 活跃参数 ~13B

EP 配置 (8 GPU):
GPU0: E0    GPU1: E1    ...    GPU7: E7
每个 token 只与 2 个 GPU 通信 → 通信量可控
```

### DeepSeek-V3

```text
配置: 256 专家, TopK=8 + 1 共享专家
特点:
- 细粒度专家分割 (256 个小专家)
- 共享专家 (1 个，所有 token 都经过)
- 无辅助损失路由 (用动态偏置保证负载均衡)
- 大规模 EP: 256 专家分布在数百张 GPU 上
```

## 通信代价分析

EP 每层通信次数：**2 次 All-to-All**（dispatch + combine）

```text
通信量 (每 token 每层):
- Dispatch: K × d_model 字节 (K 个专家各发送一份 hidden state)
- Combine:  K × d_model 字节 (K 个专家的输出送回来)

总计: 2K × d_model 字节 / token / layer

对比:
- TP: 每 token 约 2 × d_model × 4 字节 (reduce 操作)
- EP: 每 token 约 2K × d_model × 2 字节 (A2A 操作)

当 K 较小时 (如 K=1 或 2)，EP 的通信量是可控的。
但当 K 较大时 (如 K=8)，通信量随之线性增长。
```

## 总结

```text
EP 三问:

1. 切了什么？ → 切 Expert，每个 GPU 持有部分专家参数
2. 通信发生在哪里？ → Dispatch (token → expert GPU) 和 Combine (result → token GPU)
   都是 All-to-All 通信
3. 省了什么资源，又引入了什么代价？
   - 省: 每 token 计算量 = K/N × 总 FFN 计算，参数量与计算量解耦
   - 代价: 2 次 All-to-All 通信 + Router 计算开销 + 负载均衡的复杂性
```
