# 精度

GPU/AI 精度知识总览，覆盖数据格式与量化方法。

## 阅读路线

1. **数据格式**：FP8 → FP4 → FP16/BF16/TF32 — 理解各格式的位宽、范围和精度
2. **量化方法**：线性量化 → AWQ → SmoothQuant → k-means — 掌握主流模型压缩方案

---

## 1. 数据格式 (format/)

| 格式 | 位宽 | 指数位 | 尾数位 | 典型用途 | 引入架构 |
|------|------|--------|--------|---------|---------|
| FP8 (E5M2) | 8 | 5 | 2 | 推理 | Hopper |
| FP8 (E4M3) | 8 | 4 | 3 | 训练 | Hopper |
| [FP4](./format/fp4/README.md) | 4 | 2 | 1 | 推理 | Blackwell |
| FP16 | 16 | 5 | 10 | 训练/推理 | Pascal+ |
| BF16 | 16 | 8 | 7 | 训练 | Ampere+ |
| TF32 | 19 | 8 | 10 | 训练（替代 FP32） | Ampere+ |

详见：[FP8](./format/fp8/README.md) | [FP4](./format/fp4/README.md)

## 2. 量化方法 (quant/)

| 方法 | 思路 | 文件 |
|------|------|------|
| 线性量化 | Symmetric/Asymmetric，scale + zero-point | [线性量化](./quant/linearQuant/README.md) |
| AWQ | 激活感知的仅权重量化，保护重要通道 | [AWQ](./quant/AWQ/README.md) |
| SmoothQuant | 通过缩放迁移量化难度：激活→权重 | [SmoothQuant](./quant/smooth/README.md) |
| QAT | 训练时模拟量化，反向传播校正 | [QAT](./quant/QAT/README.md) |
| k-means | 聚类权重到 2^n 个质心，存储索引 | [k-means](./quant/kmeans/README.md) |
| WNAM | n-bit 权重 + m-bit 激活组合 | [WNAM](./quant/WNAM/README.md) |

## 3. 相关主题

- 架构支持：[Hopper 架构](../01-cuda/hardware/hopper.md) (FP8/Transformer Engine)、[Blackwell 架构](../01-cuda/hardware/blackwell.md) (FP4)
- TensorRT 量化：[TensorRT](../03-llm/inference/TensorRT/README.md) — 推理引擎级别的量化部署
- 性能分析：[Roofline 模型](../09-profile/cuda/roofline.md) — 精度切换如何影响吞吐和带宽
