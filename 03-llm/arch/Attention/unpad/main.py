import torch
import torch.nn as nn
import torch.nn.functional as F

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

class VarlenAttentionTorch(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x_packed, cu_seqlens):
        """
        Args:
            x_packed: [Total_Tokens, Embed_Dim] (无 Padding 的紧凑数据)
            cu_seqlens: [Batch+1] (累积长度索引)
        """
        total_tokens, _ = x_packed.shape
        
        # 1. QKV 投影 (这是最高效的一步，完全没有 Padding 浪费)
        # [Total_Tokens, 3 * Embed_Dim]
        qkv = self.qkv_proj(x_packed)
        
        # 2. 拆分 Heads
        # [Total_Tokens, 3, Num_Heads, Head_Dim]
        qkv = qkv.view(total_tokens, 3, self.num_heads, self.head_dim)
        
        # [Num_Heads, Total_Tokens, Head_Dim] -> 这种维度排列是为了配合 SDPA
        q = qkv[:, 0].transpose(0, 1) 
        k = qkv[:, 1].transpose(0, 1)
        v = qkv[:, 2].transpose(0, 1)

        # 3. 构造 Mask (纯 Torch Varlen 的代价)
        # 注意：这一步会生成 (Total_Tokens ^ 2) 大小的 bool 矩阵
        # 如果 total_tokens > 10000，这里显存会爆炸！
        attn_mask = create_block_diagonal_mask(cu_seqlens, total_tokens, dtype=torch.bool)
        
        # [1, Total_Tokens, Total_Tokens] -> 广播到所有 Head
        attn_mask = attn_mask.unsqueeze(0) 

        # 4. 计算 Attention
        # PyTorch 2.0+ 的 scaled_dot_product_attention 会自动选择最优路径
        # 这里的 attn_mask 既屏蔽了不同句子，也起到了 Causal Mask 的作用(如果需要)
        # 此时它是 "Block Sparse" 的，但 PyTorch SDPA 仍会把它当稠密矩阵算 (除非用 NestedTensor)
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=0.0
        )
        
        # 5. 还原形状
        # [Num_Heads, Total_Tokens, Head_Dim] -> [Total_Tokens, Num_Heads, Head_Dim]
        output = output.transpose(0, 1).contiguous()
        # [Total_Tokens, Embed_Dim]
        output = output.view(total_tokens, self.embed_dim)
        
        return self.o_proj(output)

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟数据: Batch=2, 句子1长度=3, 句子2长度=2
    # Total Tokens = 5
    embed_dim = 16
    num_heads = 4
    
    # 构造 Packed Input
    x_packed = torch.randn(5, embed_dim)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    
    model = VarlenAttentionTorch(embed_dim, num_heads)
    out = model(x_packed, cu_seqlens)
    
    print(f"Input shape: {x_packed.shape}") # [5, 16]
    print(f"Output shape: {out.shape}")     # [5, 16]
    
    # 验证 Mask 是否正确
    mask = create_block_diagonal_mask(cu_seqlens, 5)
    print("\n生成的 Block Diagonal Mask:")
    print(mask.int())