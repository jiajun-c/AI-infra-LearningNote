# 多模态模型 Infra

多模态模型（图文/视频/音频）在推理和训练侧与纯语言模型有本质差异，本目录聚焦 **infra 层面**的学习。

---

## 目录结构

```
03-multi/
├── arch/                  # 多模态模型架构
│   ├── vit/               # Vision Transformer
│   ├── clip/              # 对比学习对齐
│   ├── vae/               # 变分自编码器（图像压缩）
│   ├── dit/               # Diffusion Transformer
│   ├── llava/             # LLaVA 系列架构
│   └── qwen-vl/           # Qwen-VL 架构
├── encode/                # 编码器
│   ├── image/             # 图像编码（ViT patch embed、分辨率适配）
│   ├── video/             # 视频编码（帧采样、时序建模）
│   └── audio/             # 音频编码（Whisper、mel-spectrogram）
├── inference/             # 推理优化
│   ├── prefill/           # 视觉 token prefill 优化
│   ├── kvcache/           # 多模态 KV Cache 管理
│   ├── token-compress/    # 视觉 token 压缩（token merging、pooling）
│   └── scheduler/         # 多模态请求调度（视觉预处理 pipeline）
├── train/                 # 训练
│   ├── data/              # 多模态数据加载（图文对、视频帧）
│   ├── pipeline/          # 训练流水线（encode 与 LLM 的梯度流）
│   └── finetune/          # 多模态微调（LoRA、冻结策略）
├── parallel/              # 并行策略
│   ├── encoder-tp/        # 视觉编码器的张量并行
│   └── disaggregate/      # encode 与 decode 分离部署
└── memory/                # 显存管理
    ├── image-cache/       # 图像特征缓存
    └── kv-offload/        # 多模态 KV offload
```

---

## 1. 模型架构 (arch/)

### 1.1 视觉编码器

- **ViT (Vision Transformer)**: [ViT 基础](./arch/vit/README.md) - patch embedding、位置编码、与 LLM 的接口
- **CLIP**: [CLIP 对比学习](./arch/clip/README.md) - 图文对齐、zero-shot 分类
- **VAE**: [变分自编码器](./arch/vae/README.md) - 图像压缩到 latent space（Stable Diffusion 基础）
- **DiT** ⚠️ TODO - Diffusion Transformer，用于图像生成

### 1.2 多模态 LLM 架构

- **LLaVA 系列** ⚠️ TODO - 视觉特征 projection 到 LLM embedding space
- **Qwen-VL** ⚠️ TODO - 动态分辨率、视频理解

### 1.3 模态对齐方式

| 方式 | 代表模型 | 特点 |
|------|---------|------|
| Cross-Attention | Flamingo | 视觉特征作为 KV |
| Projection | LLaVA | MLP 将视觉 token 投影到文本空间 |
| Q-Former | BLIP-2 | 可学习 query 压缩视觉信息 |
| 原生多模态 | GPT-4o | 统一编解码空间 |

---

## 2. 编码器 (encode/)

### 2.1 图像编码

- **分辨率适配** ⚠️ TODO - 动态分辨率（AnyRes）、图像切片策略
- **Patch Embedding** ⚠️ TODO - 不同分辨率下的 token 数量与计算量
- **预处理 Pipeline** ⚠️ TODO - CPU decode → resize → normalize → GPU 传输

### 2.2 视频编码

- **帧采样策略** ⚠️ TODO - 均匀采样、关键帧提取、temporal token 压缩
- **时序建模** ⚠️ TODO - 3D attention、时序位置编码

### 2.3 音频编码

- **Whisper 架构** ⚠️ TODO - mel-spectrogram → encoder → cross-attention
- **音频 token 化** ⚠️ TODO - codec 模型（EnCodec、SoundStream）

---

## 3. 推理优化 (inference/)

### 3.1 视觉 token 的 Prefill 优化

多模态推理的核心瓶颈：图像会产生大量视觉 token（ViT-L 下 256~1024 个），prefill 计算量远超纯文本。

- **ChunkedPrefill** ⚠️ TODO - 视觉 token 分块 prefill，与文本 token 交织
- **视觉 Prefill 并行** ⚠️ TODO - 视觉编码与 LLM prefill 的 overlap

### 3.2 KV Cache 管理

- **视觉 token KV 特点** ⚠️ TODO - 视觉 token KV 巨大但跨请求可复用
- **多模态 prefix cache** ⚠️ TODO - 相同图片的 KV 共享（vLLM/SGLang prefix cache）
- **KV offload** ⚠️ TODO - 视觉 KV offload 到 CPU/磁盘

### 3.3 视觉 token 压缩

减少进入 LLM 的 token 数量：

- **Token Merging (ToMe)** ⚠️ TODO - 相似 token 合并
- **Pooling / Resampler** ⚠️ TODO - Q-Former、average pooling
- **动态 token 数** ⚠️ TODO - 根据图像复杂度自适应 token 数

### 3.4 请求调度

- **视觉预处理异步化** ⚠️ TODO - CPU 图像解码/resize 与 GPU 推理 pipeline
- **多模态请求优先级** ⚠️ TODO - 视觉 prefill 开销预估与调度策略

---

## 4. 训练 (train/)

### 4.1 多模态数据加载

- **图文对数据集** ⚠️ TODO - WebDataset、wds 格式、多模态 DataLoader
- **视频帧采样** ⚠️ TODO - 在线 decode vs 离线预处理
- **数据混合策略** ⚠️ TODO - 图文/视频/纯文本的采样比例

### 4.2 训练流水线

- **梯度流控制** ⚠️ TODO - 冻结视觉编码器 vs 端到端训练
- **混合精度与视觉编码器** ⚠️ TODO - FP16/BF16 下的 ViT 数值稳定性
- **显存估算** ⚠️ TODO - 视觉 token 对激活值显存的影响

### 4.3 多模态微调

- **LoRA 冻结策略** ⚠️ TODO - 只训 projection 层 vs LoRA 全参数
- **指令微调数据格式** ⚠️ TODO - 图文交织的对话数据格式

---

## 5. 并行策略 (parallel/)

### 5.1 视觉编码器并行

- **编码器 TP** ⚠️ TODO - ViT 的张量并行切分方式（attention/MLP 切分）
- **编码器 DP** ⚠️ TODO - 多图独立编码的数据并行

### 5.2 编码-解码分离部署

- **disaggregated prefill** ⚠️ TODO - 视觉编码在专用节点，LLM 在 decode 节点
- **编解码带宽** ⚠️ TODO - 视觉特征传输的通信开销

---

## 6. 显存管理 (memory/)

### 6.1 图像特征缓存

- **特征复用** ⚠️ TODO - 同一图片多次出现时复用编码结果
- **缓存淘汰策略** ⚠️ TODO - LRU/LFU 策略在多模态场景

### 6.2 多模态显存分析

| 组件 | 显存来源 | 量级（7B 模型，单图 256 token） |
|------|---------|-------------------------------|
| 视觉编码器权重 | ViT-L | ~1GB |
| 视觉 token 激活 | 256 × hidden_size | ~100MB |
| LLM KV Cache | seq_len × layers × heads | 视 seq_len 而定 |
| Projection 层 | MLP 权重 | ~几十 MB |

---

## 参考资料

- LLaVA 论文：Visual Instruction Tuning
- CLIP 论文：Learning Transferable Visual Models From Natural Language Supervision
- Flamingo 论文：A Visual Language Model for Few-Shot Learning
- vLLM 多模态文档：multimodal inputs
- SGLang 多模态支持

---

## 待办事项

- [ ] `arch/dit/` - Diffusion Transformer 架构文档
- [ ] `arch/llava/` - LLaVA 系列架构文档
- [ ] `encode/image/` - 动态分辨率与 AnyRes 策略
- [ ] `encode/video/` - 视频帧采样与时序建模
- [ ] `inference/token-compress/` - ToMe 等 token 压缩方法
- [ ] `inference/kvcache/` - 多模态 prefix cache 实现
- [ ] `parallel/disaggregate/` - 编码-解码分离部署
