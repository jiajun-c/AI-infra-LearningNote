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


## 2. 使用