import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 定义两个 Attention 类 (如前文所述)
# ==========================================
class StandardAttentionNoCache(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q, k, v):
        seq_len = q.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)


class StandardAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.k_cache = None
        self.v_cache = None

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, q, k, v):
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=-2)
            self.v_cache = torch.cat([self.v_cache, v], dim=-2)

        q_len = q.shape[2]
        kv_seq_len = self.k_cache.shape[2]
        
        scores = torch.matmul(q, self.k_cache.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if q_len > 1:
            mask = torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=q.device)
            mask = torch.triu(mask, diagonal=kv_seq_len - q_len + 1)
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, self.v_cache)


# ==========================================
# 2. 模拟推理的主程序
# ==========================================
def main():
    torch.manual_seed(42)
    
    # 模型超参数
    batch_size = 1
    num_heads = 2
    head_dim = 64
    prompt_len = 4
    decode_steps = 3

    print("==================================================")
    print(" 模式 A: 不使用 KV Cache 的灾难性调用方式 (纯数学逻辑)")
    print("==================================================")
    model_no_cache = StandardAttentionNoCache(head_dim)
    
    # 模拟外部维护的完整历史 token 特征
    history_q = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    history_k = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    history_v = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    
    print(f"[Prefill] 输入序列长度: {history_q.shape[2]}")
    out = model_no_cache(history_q, history_k, history_v)
    
    for step in range(decode_steps):
        # 模拟生成了一个新 token 的 Q, K, V
        new_q = torch.randn(batch_size, num_heads, 1, head_dim)
        new_k = torch.randn(batch_size, num_heads, 1, head_dim)
        new_v = torch.randn(batch_size, num_heads, 1, head_dim)
        
        # 外部调用者必须手动拼接所有历史数据！
        history_q = torch.cat([history_q, new_q], dim=2)
        history_k = torch.cat([history_k, new_k], dim=2)
        history_v = torch.cat([history_v, new_v], dim=2)
        
        print(f"[Decode Step {step+1}] 强行拼接后传入的输入序列长度: {history_q.shape[2]} (计算量 O(N^2))")
        out = model_no_cache(history_q, history_k, history_v)


    print("\n==================================================")
    print(" 模式 B: 标准带 KV Cache 的工程调用方式")
    print("==================================================")
    model_with_cache = StandardAttention(head_dim)
    model_with_cache.clear_cache()
    
    # Prefill: 传入最初的 Prompt
    init_q = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    init_k = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    init_v = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    
    print(f"[Prefill] 输入 Q 长度: {init_q.shape[2]}, 当前 KV Cache 长度: 无")
    out = model_with_cache(init_q, init_k, init_v)
    print(f"          -> 经过计算后，模型内部 KV Cache 长度变为: {model_with_cache.k_cache.shape[2]}")
    
    for step in range(decode_steps):
        # Decoding: 每次只传入当前这 1 个新生成的 token！
        new_q = torch.randn(batch_size, num_heads, 1, head_dim)
        new_k = torch.randn(batch_size, num_heads, 1, head_dim)
        new_v = torch.randn(batch_size, num_heads, 1, head_dim)
        
        print(f"[Decode Step {step+1}] 输入 Q 长度: {new_q.shape[2]} (计算量 O(N))")
        out = model_with_cache(new_q, new_k, new_v)
        print(f"          -> 模型内部自动拼接，当前 KV Cache 长度变为: {model_with_cache.k_cache.shape[2]}")

if __name__ == "__main__":
    main()