import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# MultiHeadAttention with KV Cache (your code)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, causal_mask=None, past_key_value=None, use_cache=False):
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)

        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # Reshape and transpose for multi-head: [B, L, H*D] -> [B, H, L, D]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate with past Key/Value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        new_past_key_value = (key, value) if use_cache else None

        # Scaled dot-product attention
        print(f"q {query.shape} key {key.shape} value {value.shape}")
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask (if provided)
        if causal_mask is not None:
            # causal_mask shape: [1, 1, curr_len, total_len]
            attn_scores = attn_scores + causal_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, value)

        # Merge heads: [B, H, L, D] -> [B, L, H*D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_linear(output)

        return (output, new_past_key_value) if use_cache else output


# ----------------------------
# Simple Decoder Layer
# ----------------------------
class SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)

    def forward(self, x, causal_mask=None, past_key_value=None, use_cache=False):
        residual = x
        attn_out = self.attn(x, causal_mask, past_key_value, use_cache)
        if use_cache:
            attn_out, new_past = attn_out
        else:
            new_past = None
        x = self.norm1(residual + attn_out)
        return (x, new_past) if use_cache else x


# ----------------------------
# Helper: Create Causal Mask
# ----------------------------
def create_causal_mask(seq_len, total_len, device):
    """
    Create a causal mask for current sequence length `seq_len` 
    against total context length `total_len`.
    Output shape: [1, 1, seq_len, total_len]
    """
    mask = torch.ones(seq_len, total_len, device=device)
    mask = torch.triu(mask, diagonal=total_len - seq_len + 1)
    mask = mask.masked_fill(mask == 1, float('-1e9'))
    return mask.unsqueeze(0).unsqueeze(0)


# ----------------------------
# Demo: Autoregressive Generation with KV Cache
# ----------------------------
def demo_kv_cache():
    torch.manual_seed(42)
    device = 'cpu'
    hidden_size = 32
    num_heads = 4
    vocab_size = 100

    # Build model
    layer = SimpleDecoderLayer(hidden_size, num_heads)
    lm_head = nn.Linear(hidden_size, vocab_size)
    embedding = nn.Embedding(vocab_size, hidden_size)

    # Simulate input tokens: [10, 20, 30]
    input_tokens = torch.tensor([[10, 20, 30]], device=device)  # [1, 3]
    embedded = embedding(input_tokens)  # [1, 3, 32]

    print("=== Step 1: Run full sequence without cache (baseline) ===")
    output_full = layer(embedded)  # [1, 3, 32]
    logits_full = lm_head(output_full)  # [1, 3, 100]
    next_token_full = logits_full[:, -1, :].argmax(dim=-1)
    print(f"Next token (no cache): {next_token_full.item()}")

    print("\n=== Step 2: Generate step-by-step WITH KV Cache ===")
    past_kv = None
    current_input = input_tokens

    # First, process the entire prompt to initialize cache
    embedded_prompt = embedding(current_input)
    _, past_kv = layer(embedded_prompt, use_cache=True)
    print(f"After processing prompt (len=3), cache length: {past_kv[0].shape[2]}")

    # Now generate one token at a time
    generated = []
    for step in range(3):  # generate 3 new tokens
        # Get last token only
        last_token_id = current_input[:, -1:]
        last_embed = embedding(last_token_id)  # [1, 1, 32]

        # Create causal mask: current len=1, total len = prompt + generated + 1
        total_len = past_kv[0].shape[2] + 1
        causal_mask = create_causal_mask(1, total_len, device)
        print(causal_mask)

        # Forward with cache
        output_step, past_kv = layer(last_embed, causal_mask=causal_mask, past_key_value=past_kv, use_cache=True)
        logits_step = lm_head(output_step)
        next_token = logits_step.argmax(dim=-1)  # [1, 1]
        generated.append(next_token.item())
        current_input = torch.cat([current_input, next_token], dim=1)

        print(f"Step {step+1}: generated token={next_token.item()}, cache length={past_kv[0].shape[2]}")

    print(f"\nGenerated sequence: {generated}")

    # Compare first generated token with full-sequence result
    print(f"\n✅ First generated token matches full-sequence result: {generated[0] == next_token_full.item()}")


# if __name__ == "__main__":
#     demo_kv_cache()
#     print(create_causal_mask(5, 2, 'cpu'))

def demo_no_kv_cache():
    # 保持种子一致以对比结果
    torch.manual_seed(42)
    device = 'cpu'
    hidden_size = 32
    num_heads = 4
    vocab_size = 100

    # 初始化模型（与 demo_kv_cache 相同的配置）
    layer = SimpleDecoderLayer(hidden_size, num_heads)
    lm_head = nn.Linear(hidden_size, vocab_size)
    embedding = nn.Embedding(vocab_size, hidden_size)

    # 初始输入: [10, 20, 30]
    input_tokens = torch.tensor([[10, 20, 30]], device=device)
    
    print("\n=== Demo: Generation WITHOUT KV Cache ===")
    
    generated = []
    current_input = input_tokens # [1, 3]

    # 我们生成 3 个新 token
    for step in range(3):
        # 1. 准备输入：必须是完整的序列 (Prompt + Past Generated)
        #    随着步骤增加，seq_len 会变成 3 -> 4 -> 5
        seq_len = current_input.size(1)
        embedded = embedding(current_input) # [1, seq_len, 32]

        # 2. 创建 Mask：这是全量的 Causal Mask
        #    seq_len 和 total_len 相等
        causal_mask = create_causal_mask(seq_len, seq_len, device)

        # 3. 前向传播：use_cache=False
        #    这里我们不传入 past_key_value，也不接收新的 past
        output = layer(embedded, causal_mask=causal_mask, use_cache=False) 
        # output shape: [1, seq_len, 32]

        # 4. 预测下一个 token
        #    我们需要取最后一个位置的输出进行预测
        logits = lm_head(output) # [1, seq_len, 100]
        next_token_logits = logits[:, -1, :] # [1, 100] 取最后一个时间步
        next_token = next_token_logits.argmax(dim=-1, keepdim=True) # [1, 1]

        # 5. 拼接结果供下一次使用
        generated.append(next_token.item())
        current_input = torch.cat([current_input, next_token], dim=1)

        print(f"Step {step+1}: Input Len={seq_len} -> Generated token={next_token.item()}")

    print(f"Final Generated sequence: {generated}")
    return generated

if __name__ == "__main__":
    # 先运行你原来的 demo
    demo_kv_cache() 
    
    # 运行无 KV Cache 版本
    res_no_cache = demo_no_kv_cache()