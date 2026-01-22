# LLM 全流程

大模型主要有下面的几个参数组成

- B：Batch Size，批次大小
- L: Seq Len，输入序列长度
- V：词表大小
- H：隐藏层大小
- N：注意力头数量
- D: Head Dim 每个头的维度，通常D=H/N
- I：Intermediate Size MLP中间层的维度

## 1. 第一阶段 Embedding

1. Tokenizer
    - 输入： B个句子
    - 操作：分词，查表，Padding到相同长度
    - 输出shape：[B, L]

2. Embedding Layer
    - 输入: [B, L]
    - 查embedding表，将每个整数ID映射为一个长度为H的向量
    - 输出shape：[B, L, H]


## 2. 第二阶段 Transformer Block 阶段

1. RMSNorm: Shape不变，仍然为 [B, L, H]

2. QKV Projection: 线性映射
    - 操作：将输入x乘以Wq, Wk, Wv
    - 逻辑shape：生成Q，K，V三个矩阵
    - 变换：[B, L, H] -> [B, L, N, D]

3. RoPE 旋转位置编码
    - 操作：在Q和K上应用位置编码
    - Shape不变

4. 将head_dim移动到前面维度， transpose，K -> KT [B, N, D, L]

5. Q @ KT = [B, N, L, D] * [B, N, D, L] = [B, N, L, L]

6 Softmax,形状不发生改变

7. Apply V（加权求和）Score @ V
[B, N, L, L] * [B, N, L, D] -> [B, N, L, D] = [B, L, H]

8. Output Projection

RMSNorm: shape不变，仍为 [B, L, H]

9. Residual Add(残差连接)

x + Attention(x) shape 不变 [B, L, H]

结束后开始使用 MLP/FFN

- RMSNorm: shape不变 [B, L, H]
- Gate/Up proj: 线性映射将维度从H拓展到I维度 [B, L, H] -> [B, L, I]
- SiLU：[B, L, I]
- 降低维度：[B, L, I] -> [B, L, H]
- 残差连接：x + MLP(x) -> [B, L, H]

## 3. 第三阶段 输出生成阶段，预测下一个词

1. Final RMSNorm
- shape 不变 [B, L, H]

2. LM Head(Linear Layer -> 映射到词表)

输出是到下一个词的概率
[B, L, H] -> [B, L, V]

3. 取最后一个token

[B, L, V] -> [B, 1, V]

4. Softmax & Sampling

对logits进行softmax归一化为概率，然后使用top-k/top-p采样或Greedy Seach

输入 [B, V]

输出得到下一个token的ID，输出为 [B, 1]

```shell
Input (Text List)
   ↓
Tokenizer
   ↓
Input IDs:       [B, L]
   ↓
Embedding:       [B, L, H]
   ↓
(Transformer Layers Loop)
   Attn Input:   [B, L, H]
   Q/K/V Split:  [B, L, N, D]
   Attn Score:   [B, N, L, L]  <-- 显存消耗大户 (O(L^2))
   Attn Output:  [B, L, H]
   MLP Input:    [B, L, H]
   MLP Inter:    [B, L, I]
   MLP Output:   [B, L, H]
   ↓
Final Norm:      [B, L, H]
   ↓
LM Head:         [B, L, V]     <-- 此时包含所有位置的预测
   ↓
Extract Last:    [B, 1, V]     <-- 只取最后一个时间步
   ↓
Sampling:        [B, 1]        <-- 最终输出的 Next Token ID
```