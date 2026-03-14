import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandKvCache(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, past_key_value=None, use_cache=False):
        """
        x shape: [batch_size, seq_len, embed_dim]
        past_key_value: 包含 (past_K, past_V) 的元组，用于解码阶段
        """
        batch_size, q_len, _ = x.shape

        # --- 阶段 1：计算当前输入 x 的 Q, K, V ---
        q = self.q_proj(x).view(batch_size, q_len, self.hidden_dim)
        k = self.k_proj(x).view(batch_size, q_len, self.hidden_dim)
        v = self.v_proj(x).view(batch_size, q_len, self.hidden_dim)

        # --- 阶段 2：处理 KV Cache ---
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        # 保存当前的 KV 用于下一步生成
        current_key_value = (k, v) if use_cache else None

        # 获取完整的 KV 序列长度
        kv_len = k.size(1)

        # --- 阶段 3：注意力计算 ---
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)

        print(f"q_len {q_len} kv_len {kv_len}")
        # 动态生成 Causal Mask
        # 如果 q_len > 1 (如 Prefill 阶段)，我们需要遮蔽未来信息
        # 如果 q_len == 1 (如 Decode 阶段)，当前 Token 需要看到所有历史信息，不需要遮蔽
        if q_len > 1:
            mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device), diagonal=kv_len - q_len + 1)
            print(mask)
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # --- 阶段 4：输出映射 ---
        # context = context.contiguous().view(batch_size, q_len, self.hidden_dim)
        output = self.o_proj(context)
        
        return output, current_key_value

# --- 测试代码 ---
if __name__ == "__main__":
    embed_dim = 128
    model = StandKvCache(embed_dim)
    
    # 模拟 1：Prefill 阶段 (处理初始 Prompt，长度为 4)
    prompt = torch.randn(1, 4, embed_dim)
    output_prefill, past_kv = model(prompt, use_cache=True)
    print(f"Prefill 输出形状: {output_prefill.shape}")
    print(f"Prefill 缓存形状 K: {past_kv[0].shape}, V: {past_kv[1].shape}")
    
    # 模拟 2：Decode 阶段 (生成第 5 个 Token，输入长度仅为 1)
    next_token = torch.randn(1, 1, embed_dim)
    output_decode, past_kv = model(next_token, past_key_value=past_kv, use_cache=True)
    print(f"Decode  输出形状: {output_decode.shape}")
    print(f"Decode  更新缓存 K: {past_kv[0].shape}, V: {past_kv[1].shape}")