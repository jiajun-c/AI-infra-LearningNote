# 大模型线性层组件

## 1. lm_head

lm_head又被称为是Language Modeling Head，它是整个模型的最终输出层，整个模型中只有一个，在模型的最后段

Transformer输出的是一个hidden_size形状的相连，lm_head的结构是一个线性层，其形状为(hidden_size, vocab_size)，其接受Transformer的输出将其映射到输出的词表中

```python3
lm_head = nn.Linear(hidden_size, vocab_size)
```

## 2. FFN

FFN是支持大语言模型的重要组成部分，Attention得到的是不同信息之间的关联，而FFN中则是基于在数据集训练得到的权重信息来对Attention输出的信息进行分析

FFN可以分为三步
- 升维：让模型可以看到更加复杂的细节
- 激活：过滤到无关的信息
- 降维：将处理好的信息压缩到原来的大小，方便传递给下一层