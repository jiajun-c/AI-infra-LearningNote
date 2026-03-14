import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        

        # 1. 定义 Q, K, V 的线性投影层 (利用了你刚才问到的 nn.Parameter 托管特性)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 2. 定义输出投影层
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # --- 阶段 1：线性映射与多头拆分 ---
        # 映射后 shape 变为 [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.embed_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.embed_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.embed_dim)


        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # 生成下三角 Causal Mask (阻止模型看到未来的 Token)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

        # 计算注意力权重并乘以 V
        attn_weights = F.softmax(scores, dim=-1)
        # [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
        context = torch.matmul(attn_weights, v)

        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # 最后的线性映射
        output = self.o_proj(context)
        
        return output

# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    embed_dim = 128
    num_heads = 8

    model = StandAttention(embed_dim)
    dummy_input = torch.randn(batch_size, seq_len, embed_dim)
    
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")