# pad and unpad attention

在之前的实现中我们的Attention输入往往是(batch, seq_len, hidden_dim) 这样一个三维度的数组，对于这样一个数组而言，其需要对长度不满足seq_len的输入进行pad到seq_len，某种程序上增加了这个冗余计算的量

## unpad

首先来看一个简单的pad的实现，如下所示的pad最后生成一个 （token_len x token_len）的mask，然后以这个mask去进行attention计算

```python3
def create_block_diagonal_mask(cu_seqlens, total_tokens, dtype=torch.bool):
    """
    构造一个 [Total_Tokens, Total_Tokens] 的块对角掩码。
    True 表示可以注意，False 表示屏蔽。
    """
    # 1. 计算每个句子的长度: [2, 3, 2]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = len(seqlens)
    
    # 2. 生成每个 Token 归属的 Batch ID
    # 例如: [0, 0, 1, 1, 1, 2, 2]
    # 这里的 repeat_interleave 是核心，它把 batch_idx 按句子长度展开
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=cu_seqlens.device), 
        seqlens
    )
    print(batch_ids)
    # 3. 利用广播机制生成 Mask
    # batch_ids[:, None] 是列向量 [7, 1]
    # batch_ids[None, :] 是行向量 [1, 7]
    # 相等的位置说明属于同一个句子
    mask = (batch_ids[:, None] == batch_ids[None, :])
    if True:
        # torch.tril 返回下三角矩阵 (i >= j 为 True, i < j 为 0)
        # 逻辑与操作 (&): 既要在同一个 Batch，又要满足 i >= j
        causal_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool))
        mask = mask & causal_mask
    return mask
```

假设输入为seqlen为4和5的两个

pad的话其输入为 (2, 5,...)

unpad其输入为 (9, ...)

计算量上 `2x5*5 < 9*9`