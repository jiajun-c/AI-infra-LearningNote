# Pre-training 学习计划

## Week 1：训练目标与数据

第一周的目标是先把“大模型训练到底在优化什么”讲清楚。不要急着进入并行训练和框架源码，先理解 pre-training 的输入、输出、loss、数据组织方式，以及它和 SFT / RLHF / DPO 的区别。

## 1. 核心问题

这一周需要回答下面几个问题：

```text
1. pre-training 学的是什么？
2. next token prediction 的 loss 是什么？
3. token、sequence、batch、step、epoch 分别是什么意思？
4. 为什么 LLM pre-training 通常不是“按样本分类”，而是按 token 预测？
5. pre-training / SFT / RLHF / DPO 的训练目标有什么区别？
6. 数据质量、数据量、去重、污染会怎样影响训练？
```

## 2. 学习路径

### Day 1：理解 pre-training 的任务形式

LLM pre-training 最常见的目标是 causal language modeling，也就是 next token prediction。

它的核心想法非常简单：

```text
给模型一段前文，让模型预测下一个 token。
```

比如原始文本是：

```text
I love deep learning
```

经过 tokenizer 后，假设得到：

```text
[I, love, deep, learning]
```

那么训练样本不是人工额外标注出来的，而是从文本本身自动构造：

```text
输入: [I, love, deep]
标签: [love, deep, learning]
```

这就是 self-supervised learning：标签来自原始文本自身。

#### 2.1 Causal Language Modeling

给定一段 token 序列：

```text
x = [x1, x2, x3, ..., xT]
```

模型在每个位置预测下一个 token：

```text
p(x2 | x1)
p(x3 | x1, x2)
p(x4 | x1, x2, x3)
...
p(xT | x1, ..., xT-1)
```

也就是：

```text
位置 1：看见 x1，预测 x2
位置 2：看见 x1, x2，预测 x3
位置 3：看见 x1, x2, x3，预测 x4
...
```

它叫 causal，是因为模型只能看当前位置之前的 token，不能偷看未来 token。

在 Transformer 里，这通常通过 causal mask 实现：

```text
第 i 个位置只能 attend 到 <= i 的位置
不能 attend 到 > i 的未来位置
```

如果不加 causal mask，模型训练时就能直接看到答案，loss 会虚假变低，但模型没有学会真正的自回归生成。

#### 2.2 从文本概率到 loss

语言模型本质上是在建模一段文本的概率：

```text
p(x1, x2, ..., xT)
```

根据链式法则，可以拆成：

```text
p(x1, x2, ..., xT)
= p(x1) * p(x2 | x1) * p(x3 | x1, x2) * ... * p(xT | x1, ..., xT-1)
```

pre-training 的目标是让训练语料里的文本概率尽可能高，也就是最大化 likelihood。

训练目标是最大化整段文本的似然，等价于最小化 negative log likelihood：

```text
L = - sum_t log p(x_t | x_<t)
```

为什么要取 log？

```text
原始目标: 最大化一堆概率的乘积
取 log 后: 最大化一堆 log probability 的和
再取负号: 变成最小化 loss
```

所以训练 loss 越低，表示模型给真实下一个 token 分配的概率越高。

#### 2.3 logits、softmax 和 cross entropy

模型的输出不是直接的概率，而是 logits。

假设：

```text
batch_size = B
sequence_length = T
vocab_size = V
```

那么：

```text
input_ids: [B, T]
logits:    [B, T, V]
```

含义是：

```text
logits[b, t, :] 表示第 b 条序列、第 t 个位置上，对词表里 V 个 token 的打分
```

要把 logits 变成概率，需要 softmax：

```text
p(token_id = k) = exp(logits[k]) / sum_j exp(logits[j])
```

cross entropy 做的事情是：

```text
1. 对 logits 做 log_softmax
2. 取出正确 label 对应位置的 log probability
3. 加负号
4. 对所有 token 求平均或求和
```

对于单个位置，如果正确答案 token 是 y：

```text
loss = -log softmax(logits)[y]
```

如果模型给正确 token 的概率是 0.8：

```text
loss = -log(0.8) ≈ 0.22
```

如果模型给正确 token 的概率是 0.01：

```text
loss = -log(0.01) ≈ 4.61
```

所以 cross entropy 会强烈惩罚“真实 token 概率很低”的情况。

#### 2.4 为什么 label 要右移一位

工程上通常写成：

```text
logits = model(input_ids)
loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])
```

这里的核心是 shift：

```text
input_ids[:, :-1] 作为上下文位置
input_ids[:, 1:]  作为下一个 token 的标签
```

举例：

```text
input_ids = [I, love, deep, learning]
```

模型输出：

```text
logits[0] 用来预测 love
logits[1] 用来预测 deep
logits[2] 用来预测 learning
```

所以：

```text
使用 logits[:-1]
对应 labels[1:]
```

最后一个位置的 logits 通常没有标签，因为它要预测序列外的下一个 token：

```text
logits[3] 需要预测 learning 后面的 token，但当前训练 chunk 里没有这个标签
```

所以简单实现里会丢掉最后一个 logits。

#### 2.5 PyTorch 里的形状变化

`torch.nn.functional.cross_entropy` 通常期望：

```text
input:  [N, C]
target: [N]
```

其中：

```text
N: 样本数
C: 类别数
```

对 LLM 来说，每个 token 位置都是一个分类样本，类别数就是 vocab size。

所以需要把：

```text
shift_logits: [B, T-1, V]
shift_labels: [B, T-1]
```

reshape 成：

```text
shift_logits: [B * (T-1), V]
shift_labels: [B * (T-1)]
```

典型代码：

```python
import torch.nn.functional as F

logits = model(input_ids)              # [B, T, V]

shift_logits = logits[:, :-1, :]       # [B, T-1, V]
shift_labels = input_ids[:, 1:]        # [B, T-1]

loss = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
)
```

这行代码的语义是：

```text
把 batch 内所有 token position 摊平成 B * (T-1) 个分类任务。
每个任务都是：根据当前位置 hidden state，预测下一个 token 的 id。
```

#### 2.6 一个完整小例子

假设词表只有 5 个 token：

```text
0: I
1: love
2: deep
3: learning
4: <eos>
```

文本：

```text
I love deep learning
```

token id：

```text
input_ids = [0, 1, 2, 3]
```

训练时：

```text
输入位置 0: token 0 = I        标签是 token 1 = love
输入位置 1: token 1 = love     标签是 token 2 = deep
输入位置 2: token 2 = deep     标签是 token 3 = learning
```

如果模型在位置 0 输出的概率是：

```text
p(I)        = 0.05
p(love)     = 0.80  <- 正确答案
p(deep)     = 0.05
p(learning) = 0.05
p(<eos>)    = 0.05
```

这个位置的 loss 是：

```text
-log(0.80)
```

如果位置 1 正确答案 deep 的概率是 0.6，位置 2 正确答案 learning 的概率是 0.3，那么总 loss 可以写成：

```text
loss = mean([
    -log(0.8),
    -log(0.6),
    -log(0.3),
])
```

这就是 pre-training 中最基础的一步。

#### 2.7 Day 1 要点

要点：

- 输入和标签来自同一段文本，只是标签相对输入右移一位。
- 模型不是直接“理解文本”，而是在大量文本上学习下一个 token 的条件分布。
- loss 是按 token 计算的，所以训练数据规模通常用 tokens 而不是 samples 描述。
- logits 是模型对整个词表的打分，softmax 后才是概率。
- cross entropy 本质上是在惩罚模型给真实下一个 token 的概率太低。
- causal mask 保证模型只能看过去，不能看未来。

### Day 2：理解 token、sequence、batch、step

需要区分几个概念：

```text
token: tokenizer 切分后的最小训练单位
sequence length: 每条训练序列包含多少 token
micro batch: 单次 forward/backward 的小 batch
global batch: 所有并行 worker + 梯度累积后的总 batch
step: 一次 optimizer update
epoch: 数据集被完整遍历一次
```

对于 LLM pre-training，更常见的统计方式是：

```text
训练 tokens = global_batch_tokens * optimizer_steps
```

其中：

```text
global_batch_tokens = micro_batch_size
                    * sequence_length
                    * data_parallel_size
                    * gradient_accumulation_steps
```

这也是为什么 scaling law 里会关注：

```text
模型参数量 N
训练 token 数 D
训练 FLOPs C
```

### Day 3：理解数据管线

pre-training 数据通常来自大量文本语料，例如网页、代码、书籍、论文、百科、论坛等。

一个典型数据管线：

```text
raw text
-> 清洗
-> 去重
-> 质量过滤
-> 安全/隐私过滤
-> tokenizer
-> packing / chunking
-> dataloader
-> model training
```

重点理解：

- 去重会影响模型是否反复记忆相同内容。
- 质量过滤会影响模型最终能力上限。
- benchmark 泄漏会让评测结果虚高。
- packing 可以减少 padding 浪费，提高 token 利用率。
- 数据配比会影响模型能力分布，例如代码、数学、多语言、百科知识。

### Day 4：对比 pre-training、SFT、RLHF、DPO

四个阶段的目标不同：

| 阶段 | 数据形式 | 目标 |
| --- | --- | --- |
| Pre-training | 大规模自然文本 token 序列 | 学习 next token distribution |
| SFT | instruction-response 样本 | 学会按指令回答 |
| RLHF | prompt + 人类偏好/奖励模型 | 让输出更符合人类偏好 |
| DPO | chosen / rejected response pair | 直接优化偏好差异 |

直观理解：

```text
Pre-training: 学语言和知识的底座
SFT: 学会对话和遵循指令
RLHF: 学会更偏好人类喜欢的回答
DPO: 用更简单的方式做偏好对齐
```

注意：

- Base 模型通常主要来自 pre-training。
- Instruct 模型通常是在 base 模型上经过 SFT / RLHF / DPO 等对齐阶段。
- pre-training 决定了模型的大部分基础能力，对齐阶段更多改变输出风格和行为偏好。

### Day 5：做一次 token 和 loss 的手算

用一小段文本手动模拟 causal LM 的训练样本：

```text
文本: "I love deep learning"
tokens: [I, love, deep, learning]
input:  [I, love, deep]
label:  [love, deep, learning]
```

要能解释：

```text
第 1 个位置：用 I 预测 love
第 2 个位置：用 I love 预测 deep
第 3 个位置：用 I love deep 预测 learning
```

并理解 loss：

```text
loss = -log p(love | I)
     + -log p(deep | I, love)
     + -log p(learning | I, love, deep)
```

这一步很重要，因为后面的显存、FLOPs、并行训练，本质都是围绕这个计算图扩展。

## 3. 本周阅读入口

建议按下面顺序阅读仓库已有笔记：

```text
concept/README.md
03-llm/train/dataset/README.md
03-llm/train/finetuning/SFT/README.md
03-llm/train/finetuning/RLHF/README.md
03-llm/train/finetuning/DPO/README.md
011-train/scalingLaw/README.md
```

读的时候重点关注：

```text
pre-training 和 SFT 的 loss 是否一样？
RLHF 和 DPO 为什么需要偏好数据？
训练 token 数为什么比 epoch 更常被讨论？
Chinchilla 为什么说很多大模型 under-trained？
```

## 4. 本周产出

完成这一周后，应该能写出一页总结：

```text
1. Pre-training 的目标函数是什么？
2. causal LM 的 input/label 如何构造？
3. token、sequence、batch、step 的关系是什么？
4. pre-training / SFT / RLHF / DPO 的区别是什么？
5. 为什么 scaling law 关注训练 token 数？
```

## 5. 检查清单

- [ ] 能用公式写出 next token prediction loss。
- [ ] 能解释为什么 label 是 input 右移一位。
- [ ] 能区分 micro batch、global batch、gradient accumulation。
- [ ] 能根据 batch 配置估算一次 optimizer step 消耗多少 tokens。
- [ ] 能说明 pre-training、SFT、RLHF、DPO 的数据形式和目标差异。
- [ ] 能解释为什么 Chinchilla Scaling Law 强调训练 token 数。
