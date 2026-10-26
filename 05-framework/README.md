# 框架实现

建议先理解 [PyTorch](./pytorch/README.md) 的张量、自动微分和分布式接口，再根据目标进入训练或推理框架。

| 路线 | 入口 | 阅读重点 |
| --- | --- | --- |
| 基础框架 | [PyTorch](./pytorch/README.md) | Tensor、autograd、torch.compile、DTensor、数据加载 |
| 推理服务 | [vLLM](./vllm/README.md) → [SGLang](./sglang/README.md) | 调度、KV Cache、权重加载和并行 |
| 训练系统 | [Megatron](./megtron/README.md) → [DeepSpeed](./deepspeed/README.md) → [Slime](./slime/README.md) | 并行、checkpoint、训练流程 |

机制基础：[LLM](../03-llm/README.md)、[分布式](../10-dist/README.md)、[训练](../11-train/README.md)。
