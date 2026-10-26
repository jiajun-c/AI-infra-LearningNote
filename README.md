# AI-infra-LearningNote

AI Infrastructure 学习笔记，涵盖编程基础、GPU 算子、模型、通信、分布式训练、推理框架和性能分析。目录编号是历史名称，阅读时按下面的依赖关系前进。

## 推荐阅读顺序

1. **基础**：[语言与 Kernel DSL](./02-lang/README.md) → [系统基础](./07-system/README.md) → [CUDA 与 GPU](./01-cuda/README.md)。
2. **模型**：[LLM](./03-llm/README.md) → [精度与量化](./12-prec/README.md) → [多模态](./03-multi/README.md)。
3. **训练**：[通信](./04-comm/README.md) → [分布式并行](./10-dist/README.md) → [训练流程](./11-train/README.md)。
4. **实现与验证**：[框架](./05-framework/README.md) → [工具链](./08-tools/README.md) → [性能分析](./09-profile/README.md)。
5. **专题延伸**：[Agent](./06-agent/README.md)、[算子开发范式](./dao/README.md)。

每个一级目录的 README 是该主题的入口；已有目标时可直接从目录地图进入。

## 目录地图

| 阶段 | 目录 | 内容 |
| --- | --- | --- |
| 基础 | [02-lang](./02-lang/README.md) | C++、Python、Triton、CuTe DSL、CUTLASS |
| 基础 | [07-system](./07-system/README.md) | CPU、内存、操作系统、网络与设备 |
| 基础 | [01-cuda](./01-cuda/README.md) | GPU 架构、CUDA、内存、同步与算子 |
| 模型 | [03-llm](./03-llm/README.md) | 模型组件、推理、优化器与评测 |
| 模型 | [12-prec](./12-prec/README.md) | 数据格式、量化与数值稳定性 |
| 模型 | [03-multi](./03-multi/README.md) | 视觉与生成模型、多模态推理 |
| 训练 | [04-comm](./04-comm/README.md) | 集合通信、NCCL、互联与重叠 |
| 训练 | [10-dist](./10-dist/README.md) | DP、DDP、FSDP、ZeRO、TP、PP、EP |
| 训练 | [11-train](./11-train/README.md) | 预训练、后训练、损失、checkpoint |
| 实现 | [05-framework](./05-framework/README.md) | PyTorch、vLLM、SGLang、Megatron、DeepSpeed |
| 实现 | [08-tools](./08-tools/README.md) | 编译器、融合、自定义算子与工程工具 |
| 实现 | [09-profile](./09-profile/README.md) | Profiling、调试、建模与评测 |
| 延伸 | [06-agent](./06-agent/README.md) | Agent 框架、推理与向量检索 |
| 延伸 | [dao](./dao/README.md) | 算子开发范式与任务划分 |

## 目录边界

- `03-llm` 放模型结构和推理机制；`11-train` 放训练流程；`10-dist` 放跨设备并行机制。已有交叉笔记保留原位，从对应入口互相链接。
- `04-comm` 解释通信原语和库；`07-system/net` 解释系统和网络基础。
- `01-cuda` 放 CUDA 底层与算子实现；`02-lang` 放语言和 DSL；`08-tools` 放构建、编译和集成工具。
- 新笔记优先写入最具体的主题目录，并在最近的上级 README 添加入口。代码实验应记录依赖、运行方法和观察结果。

待补主题见 [TODO.md](./TODO.md)。
