import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_kv):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len,  _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        past_kv = (k, v)
        kv_len = k.size(2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if seq_len > 1:
            casual_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=q.device), 
                diagonal=kv_len - seq_len + 1
            )
            scores.masked_fill_(casual_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        return output, past_kv

if __name__ == "__main__":
    num_heads = 8
    embed_dim = 128
    model = MultiHeadAttention(num_heads, embed_dim)

    #prefill
    prompt = torch.randn(1, 4, embed_dim)
    out, kv_cache = model(prompt, None)
    print(f"out shape: {out.shape}")

    #decode
    token = torch.randn(1, 1, embed_dim)
    out, kv_cache = model(token, kv_cache)
    print(f"out shape: {out.shape}")
        