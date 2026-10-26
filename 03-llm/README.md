# LLM 模型与推理

这里关注模型组件、推理机制和模型侧实验。完整训练流程从 [训练入口](../11-train/README.md) 阅读，多设备策略从 [分布式入口](../10-dist/README.md) 阅读。

1. [深度学习基础](./foundation/dl/) → [模型结构](./arch/README.md)：从词元、位置编码到 Attention、FFN、MoE。
2. [模型计算流](./arch/flow/README.md) → [权重形状](./model/weightShape/) → [模型保存与加载](./model/save_load/README.md)。
3. [KV Cache](./inference/kvcache/README.md) → [Continuous Batching](./inference/continuousBatching/README.md) → [Chunked Prefill](./inference/chunkPrefill/README.md) → [投机解码](./inference/speculative/README.md)。
4. [评测指标](./bench/metrics/README.md) → [推理 Benchmark](./bench/InferBench/README.md)。

专题：[优化器](./optimizer/)、[模型并行示例](./parallel/)、[模型中间表示](./IR/README.md)、[强化学习](./RL/README.md)、[架构搜索](./NAS/README.md)。
