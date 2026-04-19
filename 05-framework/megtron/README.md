# megtron 架构

## 1. 整体架构

```cpp
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core - 核心组件
│   │   ├── models/              # 模型定义 (GPT, BERT, T5, MoE, 多模态)
│   │   ├── transformer/         # Transformer 基础模块
│   │   ├── tensor_parallel/     # 张量并行实现
│   │   ├── pipeline_parallel/   # 流水线并行实现
│   │   ├── distributed/         # 分布式训练 (DDP, FSDP)
│   │   ├── optimizer/           # 优化器
│   │   ├── datasets/            # 数据集
│   │   ├── inference/           # 推理引擎
│   │   └── export/              # 模型导出 (TensorRT-LLM)
│   ├── training/                # 训练脚本和工具
│   ├── legacy/                  # 遗留组件
│   ├── post_training/           # 后训练 (量化、蒸馏、剪枝)
│   └── rl/                      # 强化学习 (RLHF)
├── examples/                    # 训练示例
├── tools/                       # 工具
└── docs/                        # 文档
```

### 1.1 并行化策略

- TP：层内并行，分割矩阵运算
- PP：层间并行，分割transformer层
- DP：数据并行，复制模型
- CP：序列并行，分割长序列
- EP：MoE专家并行

