# Flash-Decoding

Flash-Decoding 是一种针对 **LLM 解码阶段** 的注意力优化技术，专门解决长序列推理时的性能瓶颈。

## 背景问题

在 LLM 推理的解码阶段，每次只生成一个 token，Q 只有 1 个位置：

```
Q: [B, H, 1, D]     # 只有当前生成的 token
K: [B, H, L, D]     # L 是历史序列长度（可能很长，如 64K、128K）
V: [B, H, L, D]
```

### 传统方法的瓶颈

标准 decoding kernel 使用单个 Block 串行遍历整个序列：

```
┌─────────────────────────────────────────────────────┐
│  Block 0 串行处理: [K₀][K₁][K₂]...[K_L]               │
│  循环 L/BLOCK_SEQ 次，无法并行                         │
└─────────────────────────────────────────────────────┘
```

当序列长度 L 增大时，单个 Block 的串行处理成为严重瓶颈。

## 核心思想

**在序列维度上并行化**：将 K/V 分成多个 chunk，每个 chunk 用独立的 Block 并行处理，最后通过全局归约合并结果。

```
Stage 1: 并行计算每个 chunk 的局部 attention
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Block 0 │  │ Block 1 │  │ Block 2 │  ... (并行执行)
│ K[0:N]  │  │ K[N:2N] │  │ K[2N:3N]│
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     ▼            ▼            ▼
  (m₀,l₀,o₀)  (m₁,l₁,o₁)  (m₂,l₂,o₂)   # 局部结果

Stage 2: 全局归约，合并所有局部结果
         ┌──────────────────┐
         │  Global Reduce   │
         │  合并所有 chunk   │
         └────────┬─────────┘
                  ▼
              最终 Output
```

## 算法流程

### Stage 1: 分块计算局部 Attention

每个 Block 独立处理一个 chunk：

```python
# 每个 (batch_head, chunk_id) 对应一个 Block
for chunk in chunks:
    # 1. 计算当前 chunk 的 Q*K^T
    qk = Q @ K_chunk.T

    # 2. 计算局部 softmax 统计量
    m_i = max(qk)              # 局部最大值
    l_i = sum(exp(qk - m_i))   # 局部分母
    o_i = sum(exp(qk - m_i) * V_chunk)  # 局部加权输出

    # 3. 写入全局内存
    store(mid_m[chunk], m_i)
    store(mid_l[chunk], l_i)
    store(mid_o[chunk], o_i)
```

### Stage 2: 全局归约

使用 Online Softmax 合并所有 chunk 的结果：

```python
global_m = -inf
global_l = 0
acc = 0

for chunk in chunks:
    m_i, l_i, o_i = load(chunk)

    # Online Softmax 合并公式
    new_global_m = max(global_m, m_i)
    alpha = exp(global_m - new_global_m)
    beta = exp(m_i - new_global_m)

    global_l = global_l * alpha + l_i * beta
    acc = acc * alpha + o_i * beta
    global_m = new_global_m

output = acc / global_l
```

## 文件说明

- `profile.py`: 完整的 Flash-Decoding 实现，包含：
  - `baseline_decode_kernel`: 标准 decoding kernel（对比基准）
  - `flash_decode_stage1_kernel`: 分块并行计算
  - `flash_decode_stage2_kernel`: 全局归约
  - 性能 benchmark 对比

## 运行方式

```bash
# 运行性能对比测试
python profile.py
```

测试配置：
- 序列长度：4K 到 128K
- Batch Size: 1
- Head 数：32
- Head Dim：128

## 与 FlashAttention 的区别

| 特性     | FlashAttention | Flash-Decoding |
|------   |---------------|----------------|
| 适用场景 | 训练 + 预填充 | **解码阶段** |
| Q 形状 | [B, H, S, D] | [B, H, 1, D] |
| 并行维度 | Q 序列维度 | K/V 序列维度 |
| 核心优化 | 分块 + SRAM 复用 | **序列切分并行** |

## 性能优势

| 序列长度 | Baseline | Flash-Decoding | 加速比 |
|---------|----------|----------------|--------|
| 4K | 基准 | ~1x | - |
| 16K | 基准 | ~2x | 2x |
| 64K | 基准 | ~4x | 4x |
| 128K | 基准 | ~8x | 8x |

> 注：序列越长，Flash-Decoding 优势越明显

## 相关技术

- **FlashAttention**: 训练阶段的注意力优化
- **PagedAttention**: KV Cache 内存管理优化
- **vLLM**: 高性能推理引擎，集成了这些优化

## 参考资料

- [Flash-Decoding Blog](https://pytorch.org/blog/flash-decoding/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)