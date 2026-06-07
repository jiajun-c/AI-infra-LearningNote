# ZeRO

`ZeRO` 的全称是 `Zero Redundancy Optimizer`。

它要解决的问题是：在普通数据并行里，每张 GPU 都会保存一份完整的训练状态，显存冗余很大。这里的训练状态主要包括三类：

- 参数 `parameters`
- 参数梯度 `parameter gradients`
- 优化器状态 `optimizer states`，例如 Adam 的 `m` 和 `v`

ZeRO 的核心思路不是改变训练数学本身，而是把这些训练状态按 rank 分片存储，避免每张卡都保存完整副本。

## 1. ZeRO-0

`ZeRO-0` 可以直接理解为普通 `DDP`：

- 参数不分片
- 梯度不分片
- 优化器状态不分片

每张卡都保存完整模型、完整梯度、完整优化器状态。

## 2. ZeRO-1

`ZeRO-1` 只分片优化器状态：

- 参数不分片
- 梯度不分片
- 优化器状态分片

这样每张卡仍然能看到完整参数和完整梯度，但不再保存完整的 Adam 状态副本。

## 3. ZeRO-2

`ZeRO-2` 分片优化器状态和梯度：

- 参数不分片
- 梯度分片
- 优化器状态分片

这是第一次把“反向传播后用于更新参数的梯度”切开。

### 3.1 这里的梯度指什么

这里说的梯度，指的是反向传播后得到的参数梯度，例如：

- `dW`
- `db`

也就是优化器 `optimizer.step()` 真正会消费的那批梯度。

不是指：

- 中间激活值
- 反传过程中的临时激活梯度 `dX`
- 推理时的任何张量

### 3.2 梯度为什么能分片

因为参数更新本身通常是逐元素独立的。

以 SGD 为例：

```text
w[i] = w[i] - lr * grad[i]
```

以 Adam 为例，第 `i` 个参数的更新只依赖：

- `w[i]`
- `grad[i]`
- `m[i]`
- `v[i]`

所以只要把同一个区间上的：

- 参数分片
- 梯度分片
- 优化器状态分片

放在同一个 rank 上，这个 rank 就能独立更新自己负责的那一段。

### 3.3 梯度是怎么分片的

先看普通 `DDP`。

假设有 4 个 rank，每个 rank 在自己的 mini-batch 上完成 `backward` 后，都会得到一份“本地 batch 对完整参数的梯度贡献”：

```text
rank0: g0
rank1: g1
rank2: g2
rank3: g3
```

普通 `DDP` 的做法是 `AllReduce(sum)`：

```text
g = g0 + g1 + g2 + g3
```

于是每个 rank 最后都保留完整梯度 `g`。

`ZeRO-2` 不再让每个 rank 都保留完整 `g`，而是使用 `ReduceScatter`：

```text
ReduceScatter(sum):
1. 先对各 rank 的本地梯度做求和
2. 再把求和后的完整梯度切成 N 片，分发给 N 个 rank
```

如果完整梯度是：

```text
g = [g_part0, g_part1, g_part2, g_part3]
```

那么最终：

- rank0 只保留 `g_part0`
- rank1 只保留 `g_part1`
- rank2 只保留 `g_part2`
- rank3 只保留 `g_part3`

所以梯度分片不是“梯度没算全”，而是：

```text
全局梯度在逻辑上仍然是完整的
但物理存储上被分散到了不同 rank 上
```

### 3.4 梯度分片后如何更新参数

这是理解 `ZeRO-2` 的关键。

假设完整参数向量是：

```text
W = [w0, w1, w2, w3, w4, w5, w6, w7]
```

4 个 rank 上把它逻辑上分成 4 段：

- rank0 负责 `[w0, w1]`
- rank1 负责 `[w2, w3]`
- rank2 负责 `[w4, w5]`
- rank3 负责 `[w6, w7]`

在 `ReduceScatter` 之后，各个 rank 拿到的梯度分片正好对应这些区间：

- rank0 拿到 `[dw0, dw1]`
- rank1 拿到 `[dw2, dw3]`
- rank2 拿到 `[dw4, dw5]`
- rank3 拿到 `[dw6, dw7]`

于是每个 rank 只更新自己负责的那一小段参数：

```text
rank0: [w0, w1] = [w0, w1] - lr * [dw0, dw1]
rank1: [w2, w3] = [w2, w3] - lr * [dw2, dw3]
rank2: [w4, w5] = [w4, w5] - lr * [dw4, dw5]
rank3: [w6, w7] = [w6, w7] - lr * [dw6, dw7]
```

所以“每个节点上不是只有一部分吗”这个问题，答案正是：

```text
对，每个节点上只有一部分
但参数更新本来就可以按这部分独立完成
```

在 `ZeRO-2` 里，参数通常仍然以完整形式参与计算，但优化器 step 已经是按 shard 分工完成的。

### 3.5 什么时候需要完整参数

要区分两个阶段：

- `forward / backward compute`
- `optimizer step`

`optimizer step` 只需要本地 shard。

但在前向和反向计算某一层时，如果采用的是数据并行语义，往往还需要该层完整参数来完成矩阵乘法。因此：

- 计算阶段可能需要完整参数视图
- 更新阶段只需要本地 shard

这也是后面 `ZeRO-3` 和 `FSDP` 会进一步优化的地方。

## 4. ZeRO-3

`ZeRO-3` 在 `ZeRO-2` 的基础上，连参数也分片：

- 参数分片
- 梯度分片
- 优化器状态分片

于是每个 rank 本地同时拥有：

- 一段参数
- 一段对应梯度
- 一段对应优化器状态

这样参数更新就完全本地化了：

```text
param_shard = param_shard - lr * grad_shard
```

或者对 Adam 来说：

```text
m_shard = beta1 * m_shard + (1 - beta1) * grad_shard
v_shard = beta2 * v_shard + (1 - beta2) * grad_shard^2
param_shard = param_shard - lr * m_shard / (sqrt(v_shard) + eps)
```

只是在真正做某层前向/反向时，通常还需要临时把这一层参数 `all-gather` 回完整视图，算完之后再释放。

## 5. 和 DDP 的对比

可以把 `DDP` 和 `ZeRO-2` 的差别记成下面这张图：

```text
DDP
local grads on each rank
    ↓
AllReduce
    ↓
every rank keeps full gradient
    ↓
every rank updates full parameter

ZeRO-2
local grads on each rank
    ↓
ReduceScatter
    ↓
each rank keeps one gradient shard
    ↓
each rank updates one parameter shard
```

## 6. 最小 demo

可以直接看 [demo.py](./demo.py)，它演示的是一个最小版 `ZeRO-2` 数据流：

```text
每个 rank 本地 backward 得到完整参数梯度贡献
          ↓
DDP: AllReduce，所有 rank 都保留完整梯度
ZeRO-2: ReduceScatter，每个 rank 只保留自己的梯度分片
          ↓
每个 rank 只更新自己负责的参数分片
```

运行方式：

```bash
python 010-dist/zero/demo.py
```

这个 demo 没有起真实多进程，而是在一个进程里模拟 4 个 rank，重点是把这几件事讲清楚：

- `backward` 产生的是参数梯度
- `AllReduce` 会让每张卡都拿到完整梯度
- `ReduceScatter` 会把求和后的全局梯度切片分给不同 rank
- 只要参数分片、梯度分片、优化器状态分片对齐，就可以正确更新

## 7. 一句话总结

- `ZeRO-1`：切优化器状态
- `ZeRO-2`：切优化器状态和梯度
- `ZeRO-3`：参数、梯度、优化器状态全切

而“梯度切片后如何更新参数”的答案是：

```text
每个 rank 只更新自己负责的参数分片
因为参数更新本来就可以按 shard 独立进行
```
