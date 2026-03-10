import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, window_size):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # SWA 核心参数：窗口大小 W
        self.window_size = window_size
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_kv=None):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ==========================================
        # 🌟 SWA 核心改变 1：KV Cache 的截断机制
        # ==========================================
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
            # 强制截断：只保留最后 window_size 个 Token 的特征！
            # 这保证了显存占用永远是一个常数，绝不会 OOM
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]
            
        past_kv = (k, v)
        kv_len = k.size(2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        
        # ==========================================
        # 🌟 SWA 核心改变 2：带状下三角 Mask 的生成
        # ==========================================
        if seq_len > 1:
            # 使用相对位置坐标来精确定位
            # q_pos: [seq_len, 1]
            q_pos = torch.arange(seq_len, device=q.device).view(-1, 1)
            # k_pos: [1, kv_len]
            k_pos = torch.arange(kv_len, device=k.device).view(1, -1)
            
            # 如果存在过去的缓存，Query 的真实物理位置需要往后偏移
            q_pos = q_pos + (kv_len - seq_len)
            
            # 条件 A：因果律，Key 的位置必须 <= Query 的位置
            causal_mask = k_pos <= q_pos
            # 条件 B：滑动窗口，Key 的位置必须 > Query的位置 - 窗口大小
            window_mask = k_pos > (q_pos - self.window_size)
            
            # 取交集：既要满足因果律，又要在窗口范围内
            valid_mask = causal_mask & window_mask
            
            # 将不符合条件的区域屏蔽
            scores.masked_fill_(~valid_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        return output, past_kv

# --- 验证 SWA 的效果 ---
if __name__ == "__main__":
    num_heads = 2
    embed_dim = 16
    window_size = 3  # 故意设置一个很小的窗口：只能看当前词 + 前面 2 个词
    
    model = SlidingWindowAttention(num_heads, embed_dim, window_size)

    print("=== SWA 行为验证 ===")
    
    # 模拟输入一长串文本 (Prefill 阶段, 长度 5 > 窗口 3)
    prompt = torch.randn(1, 5, embed_dim)
    out, kv_cache = model(prompt, None)
    
    print(f"Prefill 阶段输入长度: 5")
    print(f"生成的 KV Cache 长度: {kv_cache[0].shape[2]} (被强制截断到了 window_size = 3！)")
    
    # 模拟 Decode 阶段，输入第 6 个字
    token = torch.randn(1, 1, embed_dim)
    out, kv_cache = model(token, kv_cache)
    print(f"\nDecode 阶段输入长度: 1")
    print(f"更新后的 KV Cache 长度: {kv_cache[0].shape[2]} (依然保持为 3，像履带一样向前滚动)")