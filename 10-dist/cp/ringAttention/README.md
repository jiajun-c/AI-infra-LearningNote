# RingAttention

RingAttention是CP并行的一种常见方式，其通过在序列维度进行切分实现并行，

- 每个节点维持着Q的1/N，KV的1/N
- 然后循环N个step
  - 节点之间以Ring的形式传递KV
  - 更新out和lse
- 得到当前的seq部分对应的Out和Lse

实现代码如下所示

```python

def ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)

    out = None
    lse = None
    next_k, next_v = None, None

    old_lse = False
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_varlen_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )

            outputs = _flash_attn_varlen_forward(**params)
            if len(outputs) == 8:
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs
            if block_lse.dim() == 3:
                old_lse = True
                block_lse = flatten_varlen_lse(
                    block_lse,
                    cu_seqlens=cu_seqlens,
                )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    if old_lse:
        lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)
    else:
        lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out, lse
```

可以看到每个step都要进行通信，同时由于普通的NCCL通信会占用SM，所以其同时会影响fa的性能。

## 工程优化

尽管理论上很优秀，但是由于实际通信对计算的影响以及对out和lse的更新，ring通常无法线性scale

### 1. CopyEngine优化

为了提高ring的性能，我们可以采用Copy Engine进行通信，这个操作是不占用sm的，直接创建一片对称内存，然后使用cudaMemcpy进行D2D的拷贝即可

### 2. kernel fusion

我们可以看到`update_out_and_lse` 这里有一次对于out读和写，同时这个out是fa4的输出，所以可以将其fuse到fa4的计算中，减少一次全局的读和写

### 3. 负载均衡

Ring Attention 的负载均衡是工程实践中的核心难点。如果不处理，性能会严重退化——最快的 GPU 必须空等最慢的 GPU 完成当前 step 的计算+通信。

#### 3.1 问题根源：因果注意力下的计算不对称

从上面的代码可以看到，causal 模式下的条件判断 `step <= comm.rank`：

```python
if not causal or step <= comm.rank:
    # GPU i 只在自己的 rank >= step 时才计算
```

这意味着在 $P$ 个 GPU 的环中，持有**后面 token 的 GPU 做了更多计算**：

|Ring Step|K,V 来源|GPU 0|GPU 1|GPU 2|GPU 3|
|---|---|---|---|---|---|
|step 0|GPU 0|✓ 计算|✓|✓|✓|
|step 1|GPU 1|✗ 跳过|✓|✓|✓|
|step 2|GPU 2|✗ 跳过|✗|✓|✓|
|step 3|GPU 3|✗ 跳过|✗|✗|✓|
|**计算次数**||**1**|**2**|**3**|**4**|

GPU $P-1$ 的计算量是 GPU 0 的 $P$ 倍。GPU 0 在 step 1 之后完全空闲——它既不计算（因果跳过），也不通信（已经发完自己的 KV 了）。

非 causal 模式下则完美均衡：所有 GPU 在所有 step 都计算。

#### 3.2 解决方案 1：Striped（交错）分区

**连续分区（不均衡）**：

```text
序列: [0 ─────────────────────────────────────── s]
GPU0       [0, s/P)         ← 全是早期 token，因果下只算 1 步
GPU1            [s/P, 2s/P) ← 算 2 步
GPU2                 [2s/P, 3s/P) ← 算 3 步
GPU3                      [3s/P, s] ← 算 4 步，最忙
```

**Striped 分区（均衡）**：

```text
GPU0: token 0, P, 2P, 3P, ...    ← 混合早晚 token
GPU1: token 1, P+1, 2P+1, ...    ← 混合早晚 token
GPU2: token 2, P+2, 2P+2, ...
GPU3: token 3, P+3, 2P+3, ...
```

每个 GPU 持有的 token 均匀分布在序列各处。对于任意 GPU，其 token 中有约 $1/P$ 是序列开头（短上下文），$1/P$ 是序列末尾（长上下文），**总计算量趋于均衡**。

对 Ring Attention 的影响：striped 分区后，causal 条件 `step <= comm.rank` 不再直接对应「是否需要计算」。实现上需要在 KV 传输时附带 token 位置元信息，让每个 GPU 判断哪些 token 对需要计算。等价于将 causal 判断从 rank 级下沉到 token 级。

#### 3.3 解决方案 2：变长序列的 Token 级负载均衡

对于 varlen（变长序列，如 batch 中包含不同长度的 prompt），连续分配会导致更严重的 imbalance：

```text
Batch: [seq0: 500 tokens] [seq1: 100 tokens] [seq2: 300 tokens] [seq3: 100 tokens]
总计: 1000 tokens, P=4

连续分配（不均衡）：
GPU0: seq0[0:250]       (250 tokens)
GPU1: seq0[250:500]     (250 tokens)
GPU2: seq0[500] + seq1  (200 tokens)  ← 空闲多
GPU3: seq2 + seq3       (300 tokens)  ← 忙

每个 GPU 的 attention 计算量 ∝ s_i²（自注意力近似）
负载比 ≈ 250² : 250² : 200² : 300² ≈ 6.25 : 6.25 : 4 : 9
```

**Token 级均衡方案**：

```text
按 attention 计算量（∝ token 位置的平方和）分配到各 GPU：
GPU0: 250 tokens（seq0 前半 + 其他）
GPU1: 250 tokens
GPU2: 250 tokens
GPU3: 250 tokens
→ 每卡 s_i 均衡 → O(s_i²) 近似均衡
```

更精确的方案是考虑每条序列的 prefix 长度差异，对每条序列独立做 striped partition。

#### 3.4 解决方案 3：异步流水线与通信 Overlap

即使做到了计算负载均衡，通信也可能成为瓶颈。每个 step 的结构是：

```text
Step k: [send K_k,V_k] → [recv K_{k+1},V_{k+1}] → [compute Q @ (K_{k+1},V_{k+1})]
```

如果某个 GPU 的 `send` 因为下游 GPU 仍在计算而阻塞，整条流水线 stall。解决方案：

- **双缓冲 KV**：通信和计算使用不同的 KV buffer，让 send 和 compute 并行
- **Copy Engine 通信**（上节已述）：用 GPU 的 DMA engine 做 D2D 拷贝，不占 SM，让 SM 专做 attention 计算
- **提前发送（prefetch）**：在当前 step 计算时，提前发送下一 step 的 KV，隐藏通信延迟

#### 3.5 各方案效果对比

|方案|适用场景|实现复杂度|效果|
|---|---|---|---|
|Striped 分区|causal 训练|低（改数据加载逻辑）|因果下接近均载|
|Token 级负载均衡|varlen / 推理|中（需动态分区）|彻底消除序列长度差异|
|Copy Engine|所有场景|中（需对称内存+IPC）|通信不占 SM|
|KV prefetch|所有场景|中（双缓冲）|隐藏通信延迟|
|Striped + Copy Engine|训练+推理|高|CPU、GPU、通信全均衡|
