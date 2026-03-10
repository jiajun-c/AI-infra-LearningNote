import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Q 保持完整维度映射
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # 🚨 MQA 核心改变 1：K 和 V 只有一个头，所以输出维度不再是 embed_dim，而是单一的 head_dim
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_kv=None):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len,  _ = x.shape
        
        # Q 的处理不变，形状: [B, num_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 🚨 MQA 核心改变 2：K 和 V 在 view 时的 head 数量直接硬编码为 1
        # 形状: [B, 1, seq_len, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        # Cache 拼接逻辑完全不变
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        past_kv = (k, v)
        kv_len = k.size(2)

        # 🚨 MQA 核心改变 3：利用 PyTorch 自动广播机制完成计算
        # q: [B, num_heads, q_len, head_dim]
        # k^T: [B, 1, head_dim, kv_len]
        # 矩阵乘法会自动将 1 广播为 num_heads，结果 scores: [B, num_heads, q_len, kv_len]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        
        if seq_len > 1:
            casual_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=q.device), 
                diagonal=kv_len - seq_len + 1
            )
            scores.masked_fill_(casual_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # attn_weights: [B, num_heads, q_len, kv_len]
        # v: [B, 1, kv_len, head_dim]
        # 再次触发广播机制，context: [B, num_heads, q_len, head_dim]
        context = torch.matmul(attn_weights, v)

        # 恢复形状并输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        return output, past_kv

if __name__ == "__main__":
    num_heads = 8
    embed_dim = 128
    model = MultiQueryAttention(num_heads, embed_dim)

    print("=== MQA 显存节省验证 ===")
    
    # Prefill
    prompt = torch.randn(1, 4, embed_dim)
    out, kv_cache = model(prompt, None)
    print(f"Prefill 输出形状: {out.shape}")
    # 打印缓存形状，你会发现第二维（头数）变成了 1，而不是 8！
    print(f"Cache 形状 K: {kv_cache[0].shape}, V: {kv_cache[1].shape} -> 显存占用直接变为 1/8！")

    # Decode
    token = torch.randn(1, 1, embed_dim)
    out, kv_cache = model(token, kv_cache)
    print(f"Decode 输出形状: {out.shape}")
    print(f"Cache 形状 K: {kv_cache[0].shape}, V: {kv_cache[1].shape}")