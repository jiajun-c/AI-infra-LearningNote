# GPU性能分析

## 1. 使用nvtx

nvtx是英伟达的性能分析工具库，其通过在代码添加注释的方式，帮助排查性能问题。

如下所示

```python3
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output
```

在外层调用中，使用`nvtx.range_push` 和`nvtx.range_pop`

```python3
nvtx.range_push("warmup")
for _ in range(num_warmups):
    torch.cuda.synchronize()
    logits = model.forward(data_x)
    torch.cuda.synchronize()
    loss = cs336_utils.cross_entropy(
        einops.rearrange(logits, 'b c v -> (b c) v'),
        einops.rearrange(data_y, 'b c -> (b c)')
    )
    torch.cuda.synchronize()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    optimizer.step()
    torch.cuda.synchronize()
nvtx.range_pop()
```


## 2. 理论性能分析

### Flops 和 bandwidth

PFlops = 1e3 TFlops = 1e3 GFlops = 1e3 MFlops = 1e3 KFlops

一般GPU都会使用TFlops，带宽上回使用GB/s

### 计算强度分析

AI = FLOPS/Bytes

以online softmax算子为例，其需要对每个元素进行一次max，sub，exp，div，add的操作，整体的时间复杂度在大约在5~10Flops

访存量为 Fp16 下读写各一次，共4 Bytes/element

算术强度 (AI): 10/4 = 2.5Flops/Bytes

在H100上，其计算能力为312TFlops，带宽能力为2039GB/s

计算和访存的拐点是 312*10e3/2038 = 153Flops/Bytes

该算术强度AI远远小于访存的拐点，所以仅仅考虑访存的时间。

需要的时间为等于访存的量除以带宽，以safe softmax为例，其理论访存次数主要取决于HBM上的访存次数，即2*N

所以其时间可以计算为 2*N/bandwidth

