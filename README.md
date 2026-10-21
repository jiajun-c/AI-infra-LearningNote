# AI-infra-LearningNote

面向 AI Infrastructure 的学习笔记库，记录从 GPU 硬件与 CUDA Kernel，到大模型训练、推理、分布式通信、框架实现和性能分析的知识与实验。

这里不是一个可直接部署的单体项目，而是一套按主题组织、持续补充的技术笔记。每个主题目录中的 README 通常包含概念、实现路径、代码片段、实验现象或源码阅读线索。

## 从这里开始

| 目标 | 建议路径 |
| --- | --- |
| 理解 GPU 与 CUDA | [硬件架构](./01-cuda/hardware/README.md) → [内存系统](./01-cuda/memory/README.md) → [并行原语](./01-cuda/primitives/warp/README.md) → [CUTLASS 3.x GEMM](./01-cuda/cutlass/gemm/cutlass3.x/README.md) |
| 补齐 C++ / Python / Triton 基础 | [C++ 类型系统](./02-lang/cpp/type/README.md) → [引用与转发](./02-lang/cpp/reference/README.md) → [Python](./02-lang/python/iter/README.md) → [Triton](./02-lang/Triton/README.md) |
| 学习 CuTe DSL 与异步数据搬运 | [Layout](./02-lang/cuteDSL/layout/README.md) → [mbarrier](./02-lang/cuteDSL/mbarrier/README.md) → [Pipeline](./02-lang/cuteDSL/pipeline/README.md) → [TMA](./02-lang/cuteDSL/tma/README.md) → [DSM](./02-lang/cuteDSL/dsm/README.md) |
| 学习 LLM 核心算子与推理 | [Attention](./03-llm/arch/Attention/README.md) → [MoE](./03-llm/arch/MoE/README.md) → [KV Cache](./03-llm/inference/kvcache/README.md) → [Continuous Batching](./03-llm/inference/continuousBatching/README.md) |
| 学习分布式训练 | [分布式训练总览](./10-dist/README.md) → DP / DDP → FSDP / ZeRO → TP / PP / EP → [混合并行与显存](./10-dist/combine/README.md) |
| 理解通信与网络瓶颈 | [通信与网络](./04-comm/README.md) → [集合通信](./04-comm/collective/README.md) → [NCCL](./04-comm/CCL/NCCL/README.md) → [计算通信重叠](./04-comm/overlap/README.md) |
| 理解训练全流程与损失函数 | [预训练](./11-train/pre-training/README.md) → [交叉熵损失](./11-train/loss/cross-entropy-loss/README.md) → [SFT](./11-train/post-training/SFT/README.md) → [DPO / RLHF](./11-train/post-training/Alignment/README.md) |
| 学习多模态模型 Infra | [多模态总览](./03-multi/README.md) → [ViT](./03-multi/arch/vit/README.md) → [CLIP](./03-multi/arch/clip/README.md) → [DiT](./03-multi/arch/dit/README.md) → [DiT Cache](./03-multi/inference/dit-cache/README.md) |
| 补齐系统与运行时基础 | [CPU Cache](./07-system/cpu/x86/cache/README.md) → [NUMA](./07-system/numa/README.md) → [io_uring](./07-system/os/io_uring/README.md) → [低延时网络](./07-system/net/README.md) |
| 理解精度与量化 | [精度总览](./12-prec/README.md) → [FP8](./12-prec/format/fp8/README.md) / [FP4](./12-prec/format/fp4/README.md) → [线性量化](./12-prec/quant/linearQuant/README.md) → [AWQ](./12-prec/quant/AWQ/README.md) → [SmoothQuant](./12-prec/quant/smooth/README.md) |
| 阅读框架实现与做性能优化 | [PyTorch 概览](./05-framework/pytorch/overview/README.md) → [vLLM](./05-framework/vllm/README.md) / [SGLang](./05-framework/sglang/README.md) → [性能分析](./09-profile/README.md) |

## 目录地图

```text
AI-infra-LearningNote/
├── 01-cuda/       CUDA 编程、GPU 架构、内存、算子与 CUTLASS
├── 02-lang/       C++、Python、Triton、CuTe DSL 与底层编程基础
├── 03-llm/        LLM 架构、训练、推理、并行与评测
├── 03-multi/      多模态模型：ViT、CLIP、VAE、DiT、LDM
├── 04-comm/       通信后端、NCCL、集合通信、互联与 overlap
├── 05-framework/  PyTorch、vLLM、SGLang、Megatron、DeepSpeed
├── 06-agent/      Agent 框架、推理与向量检索
├── 07-system/     CPU / 内存 / OS / 网络与各类 XPU 加速器
├── 08-tools/      编译器、算子融合、工程工具与第三方库
├── 09-profile/    性能分析、调试、建模与评测
├── 10-dist/       分布式并行、SPMD、同步与锁
├── 11-train/      预训练、后训练、损失函数、checkpoint 与 scaling law
├── 12-prec/       数值精度与量化：FP8 / FP4、AWQ、SmoothQuant 等
└── dao/           算子开发范式与任务划分
```

## 按主题导航

### CUDA 与 GPU 编程

- 硬件与执行模型：[GPU 硬件](./01-cuda/hardware/README.md)、[Hopper](./01-cuda/hardware/hopper.md)、[Blackwell](./01-cuda/hardware/blackwell.md)、[Stream](./01-cuda/stream/README.md)、[同步机制](./01-cuda/sync/README.md)
- 内存与数据搬运：[内存模型](./01-cuda/memory/README.md)、[全局内存合并访问](./01-cuda/memory/global/README.md)、[TMA](./01-cuda/memory/tma/README.md)、[Bank Conflict](./01-cuda/memory/bank/README.md)、[Pinned Memory](./01-cuda/memory/pinMemory/README.md)
- Kernel 与库：[并行算法](./01-cuda/algorithm/README.md)、[Warp 原语](./01-cuda/primitives/warp/README.md)、[CUTLASS](./01-cuda/cutlass/gemm/cutlass3.x/README.md)、[Driver API](./01-cuda/driver/README.md)

### 语言与 Kernel DSL

- C++：[类型系统](./02-lang/cpp/type/README.md)、[引用与完美转发](./02-lang/cpp/reference/README.md)、[模板](./02-lang/cpp/template/README.md)、[线程](./02-lang/cpp/thread/README.md)、[智能指针](./02-lang/cpp/point/README.md)
- Python：[迭代器](./02-lang/python/iter/README.md)、[生成器](./02-lang/python/yield/README.md)、[asyncio](./02-lang/python/async/README.md)、[类系统](./02-lang/python/class/README.md)
- Triton：[基础](./02-lang/Triton/basic/README.md)、[Matmul](./02-lang/Triton/matmul/README.md)、[FlashAttention](./02-lang/Triton/flashAttention/README.md)、[Autotune](./02-lang/Triton/autotune/README.md)
- CuTe DSL：[Layout](./02-lang/cuteDSL/layout/README.md)、[mbarrier](./02-lang/cuteDSL/mbarrier/README.md)、[Pipeline](./02-lang/cuteDSL/pipeline/README.md)、[TMA](./02-lang/cuteDSL/tma/README.md)、[DSM](./02-lang/cuteDSL/dsm/README.md)、[Tile Scheduler](./02-lang/cuteDSL/scheduler/README.md)

### LLM、分布式与通信

- 模型：[模型数据流](./03-llm/arch/flow/README.md)、[Attention](./03-llm/arch/Attention/README.md)、[MoE](./03-llm/arch/MoE/README.md)、[Linear](./03-llm/arch/Linear/README.md)、[Norm](./03-llm/arch/Norm/README.md)、[位置编码](./03-llm/arch/position_encode/relative/README.md)
- 推理：[KV Cache](./03-llm/inference/kvcache/README.md)、[Prefix Cache](./03-llm/inference/prefix_cache/README.md)、[Chunked Prefill](./03-llm/inference/chunkPrefill/README.md)、[Speculative Decoding](./03-llm/inference/speculative/README.md)、[剪枝](./03-llm/inference/prune/channel-based/README.md)
- 并行：[DP / DDP / FSDP / ZeRO](./10-dist/README.md)、[Tensor Parallel](./03-llm/parallel/TP/README.md)、[Pipeline Parallel](./10-dist/PP/README.md)、[Expert Parallel](./10-dist/ep/README.md)、[Sequence Parallel](./10-dist/cp/README.md)、[混合并行与显存](./10-dist/combine/README.md)、[显存占用](./10-dist/memory/README.md)
- 通信：[集合通信](./04-comm/collective/README.md)、[NCCL](./04-comm/CCL/NCCL/README.md)、[GIN（GPU 发起网络）](./04-comm/CCL/NCCL/gin/README.md)、[NVLink / NVSwitch](./04-comm/nvlink/README.md)、[拓扑](./04-comm/topo/README.md)、[Overlap](./04-comm/overlap/README.md)

### 框架、训练与性能

- 框架：[PyTorch](./05-framework/pytorch/README.md)、[torch.compile](./05-framework/pytorch/compile/README.md)、[vLLM](./05-framework/vllm/README.md)、[SGLang](./05-framework/sglang/README.md)、[Megatron-LM](./05-framework/megtron/README.md)、[DeepSpeed](./05-framework/deepspeed/README.md)
- 训练：[预训练](./11-train/pre-training/README.md)、[训练 step 组成](./11-train/pretrain-step/README.md)、[梯度累加](./11-train/gradAccStep/README.md)、[交叉熵损失](./11-train/loss/cross-entropy-loss/README.md)、[后训练总览](./11-train/post-training/Alignment/README.md)、[SFT](./11-train/post-training/SFT/README.md)、[Checkpoint](./11-train/checkpoint/README.md)、[Scaling Law](./11-train/scalingLaw/README.md)
- 性能：[CUDA Profiling](./09-profile/cuda/README.md)、[Roofline / FLOPs](./09-profile/cuda/theory.md)、[Warp Stall](./09-profile/cuda/stall.md)、[性能建模](./09-profile/modeling/README.md)、[调试](./09-profile/debug/README.md)

### 多模态模型

- 架构：[ViT](./03-multi/arch/vit/README.md)、[CLIP](./03-multi/arch/clip/README.md)、[VAE](./03-multi/arch/vae/README.md)、[DiT](./03-multi/arch/dit/README.md)、[LDM](./03-multi/arch/ldm/README.md)、[TextEncoder](./03-multi/arch/textEncoder/README.md)
- 推理：[DiT Cache](./03-multi/inference/dit-cache/README.md)、[Text2X](./03-multi/inference/t2x/README.md)
- 总览：[多模态模型 Infra](./03-multi/README.md)

### 精度与量化

- 数据格式：[FP8](./12-prec/format/fp8/README.md)、[FP4](./12-prec/format/fp4/README.md)
- 量化方法：[线性量化](./12-prec/quant/linearQuant/README.md)、[AWQ](./12-prec/quant/AWQ/README.md)、[SmoothQuant / 激活值平滑](./12-prec/quant/smooth/README.md)、[k-means 量化](./12-prec/quant/kmeans/README.md)、[QAT](./12-prec/quant/QAT/README.md)、[WNAM](./12-prec/quant/WNAM/README.md)
- 数值稳定性：[Batch Invariance](./12-prec/batch-invariance/README.md)

### 系统与运行时

- CPU 与缓存：[x86 Cache](./07-system/cpu/x86/cache/README.md)、[伪共享](./07-system/cpu/falseShare/README.md)、[CPU 调度](./07-system/cpu/sched/README.md)、[多线程](./07-system/cpu/multiThread/README.md)、[无锁数据结构](./07-system/cpu/spsc/README.md)
- 内存与设备：[系统访存流程](./07-system/memory/README.md)、[NUMA](./07-system/numa/README.md)、[Cache 一致性](./07-system/cache/coherent/README.md)、[设备信息 / BDF](./07-system/device/README.md)
- OS 与系统调用：[io_uring](./07-system/os/io_uring/README.md)、[epoll](./07-system/os/epoil/README.md)、[select](./07-system/os/select/README.md)、[AIO](./07-system/os/aio/README.md)、[IPC](./07-system/os/ipc/README.md)、[内核态](./07-system/os/kernel-mode/README.md)、[进程](./07-system/process/README.md)
- 网络与加速器：[低延时网络](./07-system/net/README.md)、[Kernel Bypass](./07-system/net/kernel-bypass/README.md)、[GPU 拓扑](./07-system/gpu/topo/README.md)、[NPU](./07-system/npu/README.md)、[ARM SME](./07-system/matrixUnit/arm_sme/README.md)

### 工程工具与 Agent

- 工具链：[算子融合](./08-tools/fuse/README.md)、[Torch 自定义算子](./08-tools/glue/torch/README.md)、[TVM](./08-tools/compiler/tvm/README.md)、[Einops](./08-tools/third_party/einops/README.md)、[pyproject 项目管理](./08-tools/pyproject/README.md)、[微调工具](./08-tools/tuning/README.md)
- Agent：[LangChain](./06-agent/langchain/README.md)、[推理能力](./06-agent/infer/README.md)、[向量数据库](./06-agent/vectorDB/README.md)
- 算子开发：[范式](./dao/README.md)、[任务划分](./dao/partition/README.md)

## 阅读与维护约定

- 根 README 只提供稳定入口；具体概念、源码路径和实验结论优先写在相应主题目录中。
- 新增主题时，优先补充该目录的 README，并从一个已有的上级入口建立链接。
- 代码示例应说明依赖、运行方式和预期现象；实验记录应保留关键环境与结论。
- 本仓库按知识主题组织，不保证所有目录均可独立编译或直接运行。

待补主题见 [TODO.md](./TODO.md)。

最后更新：2026-10-21
