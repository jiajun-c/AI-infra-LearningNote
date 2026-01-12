# KVcache

在大模型的推理过程中，其逻辑是根据已经传入的token来预测下一个token，因此，对于已经传入的token，我们可以将其存储起来，下次传入的时候，可以先从缓存中获取，从而提高推理速度。

在没有kvcache的时候，每次传入的token长度是不断增加的，如下所示，seq_len的长度每次+1

每次的计算量为 (seq_len * seq_len * head_dim), 对与kvcache而言，其将之前计算得到的k和v进行了缓存，使得其计算量为 (1 * seq_len * head_dim)，随着seq_len的增加，其计算量增加较少。

存储开销上，kvcache的存储开销为 (2*sizeof(type) * n_layer * d_model * seq_len * batch_size)，对于原本的模型而言，其只需要存储当前激活的显存即可。所以kvcache也是一个显存杀手

```cpp
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
```

```cpp
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

```

> 为什么不需要q cache：因为在推理中，本身就只需要用到当前的q，而不会用到历史的q