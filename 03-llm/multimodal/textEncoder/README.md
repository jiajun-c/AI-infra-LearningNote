# TextEncoder

TextEncoder的作用是将文本（token序列）编码为高维语义向量表示，供多模态模型的跨模态对齐或融合使用。

## 核心功能

在多模态模型（如CLIP、BLIP、LLaVA等）中，TextEncoder负责：

1. **文本语义提取**：将输入文本映射到语义嵌入空间，捕捉语义信息
2. **跨模态对齐**：与ImageEncoder产生的视觉特征对齐，实现文本-图像语义匹配
3. **条件生成**：在生成模型中，TextEncoder的输出作为条件信号指导图像/视频生成

## 常见架构

### Transformer-based TextEncoder

绝大多数多模态模型使用预训练的Transformer作为TextEncoder：

| 模型 | TextEncoder | 特点 |
|------|------------|------|
| CLIP | ViT-style Transformer | 对比学习预训练，文本和图像共享嵌入空间 |
| BLIP/BLIP-2 | BERT-based | 支持双向注意力，适合理解任务 |
| Stable Diffusion | CLIP ViT-L / T5 | 文本条件控制图像生成 |
| LLaVA | LLaMA/Vicuna | 直接使用LLM作为文本理解模块 |
| Flamingo | Chinchilla | 通过交叉注意力融合视觉特征 |

### 编码流程

```
Input Text
    ↓
Tokenization（BPE/WordPiece）
    ↓
Token Embedding + Positional Encoding
    ↓
Transformer Layers（Multi-Head Self-Attention + FFN）
    ↓
[CLS] token 或 平均池化
    ↓
Text Embedding（维度通常为512/768/1024）
```

## 与ImageEncoder的对齐方式

### 对比学习（Contrastive Learning）

CLIP采用对比学习对齐文本和图像嵌入：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(t_i, v_j)/\tau)}$$

其中 $\tau$ 为可学习的温度参数，$\text{sim}$ 为余弦相似度。

### Q-Former（BLIP-2）

BLIP-2引入轻量级Q-Former桥接冻结的ImageEncoder和LLM：

```
Image Features（冻结ViT）
        ↓
Q-Former（可学习查询向量 + 交叉注意力）
        ↓
LLM（冻结或微调）← Text Input
```

### Cross-Attention Fusion（Flamingo）

将视觉特征通过交叉注意力插入LLM的每个Transformer层：

```python
# 简化示意
def flamingo_layer(text_tokens, image_features):
    # Self-attention on text
    x = self_attn(text_tokens)
    # Cross-attention with image
    x = cross_attn(x, image_features)
    x = ffn(x)
    return x
```

## 关键设计考量

### 1. 冻结 vs 微调

- **冻结TextEncoder**：保留预训练知识，减少训练成本（BLIP-2策略）
- **端到端微调**：更好适配下游任务，但需要更多数据和算力

### 2. 文本长度限制

CLIP的TextEncoder通常限制77个token（受位置编码限制），对长文本描述处理能力有限。T5等模型支持更长序列。

### 3. 多语言支持

标准CLIP以英文为主，多语言场景需要使用mCLIP或专门的多语言TextEncoder。

## 参考

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
