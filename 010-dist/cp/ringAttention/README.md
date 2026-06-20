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
