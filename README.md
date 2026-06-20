# AI-infra-LearningNote

AI 基础设施学习笔记，聚焦 GPU 编程、LLM 训练与推理、通信系统、深度学习框架和性能分析。

这个仓库不是一个单一软件项目，而是一个持续演进的知识库：每个目录对应一个主题，README 负责解释核心概念、源码链路、实验代码或性能现象。

---

## 如何使用

如果你刚开始看，可以按下面的路线进入：

1. **GPU / CUDA Infra**：从 [CUDA 硬件架构](./01-cuda/hardware/README.md) 开始，再看 [内存层次与合并访问](./01-cuda/memory/global/README.md)、[TensorCore](./01-cuda/tensorCore/README.md)、[CUTLASS / CuTe](./01-cuda/cutlass/gemm/cutlass3.x/README.md)。
2. **LLM Training Infra**：先看 [Attention](./03-llm/arch/Attention/README.md)、[MoE](./03-llm/arch/MoE/README.md)，再进入 [TP](./03-llm/parallel/TP/README.md)、[PP](./03-llm/parallel/PP/README.md)、[FSDP](./03-llm/parallel/FSDP/README.md)。
3. **LLM Inference Infra**：从 [KV Cache](./03-llm/inference/kvcache/README.md)、[Continuous Batching](./03-llm/inference/contiuousBatching/README.md)、[Chunked Prefill](./03-llm/inference/chunkPrefill/README.md) 和 [FlashDecode](./03-llm/inference/flashDecode/README.md) 入手。
4. **Framework / Serving**：看 [PyTorch 架构](./05-framework/pytorch/overview/README.md)、[torch.compile](./05-framework/pytorch/compile/README.md)、[vLLM](./05-framework/vllm/README.md)、[SGLang 权重加载](./05-framework/sglang/weightLoad/README.md)。
5. **Training Compute / Scaling Law**：看 [Chinchilla Scaling Law](./011-train/scalingLaw/README.md)，理解固定训练算力下参数量和训练 token 数的分配。
6. **通信、网络与系统调优**：看 [通信与网络](./04-comm/README.md)、[NCCL](./04-comm/CCL/NCCL/README.md)、[系统与硬件](./07-system/README.md)、[CUDA Profiling](./09-profile/cuda/README.md)。

---

## 目录地图

```text
AI-infra-LearningNote/
├── 01-cuda/       CUDA 编程、GPU 架构、算子、CUTLASS/CuTe
├── 02-lang/       C++、Python、Triton 与底层编程语言基础
├── 03-llm/        LLM 架构、训练、推理、量化、并行与评测
├── 03-multi/      多模态模型 Infra，含 ViT/CLIP/VAE/DiT/LDM
├── 04-comm/       通信后端、NCCL、集合通信、网络栈与计算通信重叠
├── 05-framework/  PyTorch、vLLM、SGLang、Megatron、DeepSpeed
├── 06-agent/      Agent 框架与向量检索
├── 07-system/     CPU/GPU/NPU、内存系统、OS I/O、网络、进程模型
├── 08-tools/      编译器、项目管理、第三方库与工程工具
├── 09-profile/    性能分析、调试、优化方法与评测工具
├── 010-dist/      分布式训练专题：DP/DDP/FSDP/HSDP/ZeRO/CP
├── 011-train/     训练算力、Scaling Law、Pre/Post-Training
├── concept/       pre-training / SFT / RL 等基础概念
├── cuda/          CUTLASS / CuTe 实践代码
└── dao/           算子开发范式与任务划分
```

---

## 核心主题

### CUDA 与 GPU 编程

- 架构基础：[硬件架构](./01-cuda/hardware/README.md)、[Blackwell](./01-cuda/blackwell/README.md)、[Hopper TMA](./01-cuda/hopper/TMA/README.md)、[Hopper Pipeline](./01-cuda/hopper/pipe/README.md)
- 执行模型：[启动配置](./01-cuda/launch/README.md)、[Stream](./01-cuda/stream/README.md)、[Cooperative Groups](./01-cuda/cg/README.md)、[Warp 原语](./01-cuda/primitives/warp/README.md)
- Driver API：[总览](./01-cuda/driver/README.md)、[Stream Memory Ops](./01-cuda/driver/memory/README.md)（cuStreamWriteValue32/WaitValue32/BatchMemOp）
- 内存优化：[Bank Conflict](./01-cuda/memory/bank/README.md)、[全局内存合并](./01-cuda/memory/global/README.md)、[Cache](./01-cuda/memory/cache/README.md)、[Pin Memory](./01-cuda/pin/README.md)、[VMM](./01-cuda/memory/vmm/README.md)
- 算子实现：[HGEMV](./01-cuda/blas/hgemv/README.md)、[Element-wise](./01-cuda/op/element_wise/README.md)、[Transpose](./01-cuda/op/transpose/README.md)、[Reduce](./01-cuda/reduce/README.md)
- CUTLASS / CuTe：[CuTe 多维分块](./01-cuda/cutlass/cute/multidimTile/README.md)、[Copy](./01-cuda/cutlass/copy/README.md)、[CUTLASS 3.x GEMM](./01-cuda/cutlass/gemm/cutlass3.x/README.md)、[Device GEMM](./01-cuda/cutlass/gemm/device/README.md)

### 编程语言与 Kernel DSL

- C++：[类型系统](./02-lang/cpp/type/README.md)、[内存管理](./02-lang/cpp/memory/README.md)、[模板](./02-lang/cpp/template/README.md)、[智能指针](./02-lang/cpp/point/README.md)、[STL vector](./02-lang/cpp/stl/vector/README.md)
- Python：[迭代器协议](./02-lang/python/iter/README.md)、[yield 生成器](./02-lang/python/yield/README.md)、[asyncio](./02-lang/python/async/README.md)、[类系统](./02-lang/python/class/README.md)
- Triton：[基础语法](./02-lang/Triton/basic/README.md)、[矩阵乘法](./02-lang/Triton/matmul/README.md)、[FlashAttention](./02-lang/Triton/flashAttention/README.md)、[Autotune](./02-lang/Triton/autotune/README.md)、[Kernel Fusion](./02-lang/Triton/fusion/permuteFusion/README.md)

### LLM 架构、训练与推理

- 架构：[模型数据流](./03-llm/arch/flow/README.md)、[Attention](./03-llm/arch/Attention/README.md)、[FlashAttention V1](./03-llm/arch/Attention/FlashAttention/README.md)、[FlashAttention V2](./03-llm/arch/Attention/flashAttentionv2/README.md)、[MoE](./03-llm/arch/MoE/README.md)
- 并行训练：[DP](./010-dist/dp/README.md)、[DDP](./010-dist/DDP/README.md)、[FSDP](./010-dist/fsdp/README.md)、[HSDP](./010-dist/hsdp/README.md)、[ZeRO](./010-dist/zero/README.md)、[Distributed Transpose](./010-dist/trans/README.md)、[TP](./03-llm/parallel/TP/README.md)、[PP](./03-llm/parallel/PP/README.md)、[EP](./03-llm/parallel/EP/README.md)
- 序列并行 (CP)：[总览](./010-dist/cp/README.md)、[Megatron-SP](./010-dist/cp/Megtron-SP/README.md)、[Ring Attention](./010-dist/cp/ringAttention/README.md)、[Ulysses](./010-dist/cp/ulysses/README.md)
- 训练与微调：[Pre-Training](./011-train/pre-training/README.md)、[Post-Training SFT](./011-train/post-training/SFT/README.md)、[RLHF](./011-train/post-training/Alignment/RLHF/README.md)、[DPO](./011-train/post-training/Alignment/DPO/README.md)、[Gradient Accumulation](./011-train/gradAccStep/README.md)、[数据集处理](./03-llm/train/dataset/README.md)、[梯度检查点](./03-llm/train/LowMem/checkpoint/README.md)
- 训练算力：[Chinchilla Scaling Law](./011-train/scalingLaw/README.md)
- 推理优化：[KV Cache](./03-llm/inference/kvcache/README.md)、[Prefix Cache](./03-llm/inference/prefix_cache/README.md)、[Batching](./03-llm/inference/batch/README.md)、[Chunked Prefill](./03-llm/inference/chunkPrefill/README.md)、[Speculative Decoding](./03-llm/inference/speculative/README.md)
- 量化与压缩：[线性量化](./03-llm/inference/quant/linearQuant/README.md)、[AWQ](./03-llm/inference/quant/AWQ/README.md)、[QAT](./03-llm/inference/quant/QAT/README.md)、[SmoothQuant](./03-llm/inference/quant/smooth/README.md)、[k-means 量化](./03-llm/inference/quant/kmeans/README.md)

### 多模态 Infra

- 入口：[多模态目录](./03-multi/README.md)
- 架构：[ViT](./03-multi/arch/vit/README.md)、[CLIP](./03-multi/arch/clip/README.md)、[VAE](./03-multi/arch/vae/README.md)、[TextEncoder](./03-multi/arch/textEncoder/README.md)、[DiT](./03-multi/arch/dit/README.md)、[LDM](./03-multi/arch/ldm/README.md)
- 推理：[DiT Cache](./03-multi/inference/dit-cache/README.md)、[Text2X](./03-multi/inference/t2x/README.md)

### 通信、框架与系统

- 通信与网络：[总览](./04-comm/README.md)、[Gloo](./04-comm/backend/gloo/README.md)、[NCCL](./04-comm/CCL/NCCL/README.md)、[跨卡同步机制](./04-comm/sync/README.md)、[对称内存](./04-comm/CCL/NCCL/symm/README.md) ([PyTorch API](./05-framework/pytorch/symmMem/README.md))、[集合通信](./04-comm/collective/README.md)、[Overlap](./04-comm/overlap/README.md)、[NVLink](./04-comm/nvlink/README.md)、[NVSHMEM](./04-comm/nvshem/README.md)
- PyTorch：[Overview](./05-framework/pytorch/overview/README.md)、[Stream](./05-framework/pytorch/stream/README.md)、[Context](./05-framework/pytorch/context/README.md)、[Custom Ops](./05-framework/pytorch/custom_ops/README.md)、[Memory](./05-framework/pytorch/memory/model/README.md)
- Serving / Training Framework：[vLLM](./05-framework/vllm/README.md)、[SGLang](./05-framework/sglang/README.md)、[Megatron-LM](./05-framework/megtron/README.md)、[Slime](./05-framework/slime/README.md)、[DeepSpeed](./05-framework/deepspeed/README.md)
- 系统：[系统与硬件概述](./07-system/README.md)、[GPU](./07-system/gpu/README.md)、[GPU 内存模型](./07-system/gpu/memory/README.md)、[NPU](./07-system/npu/README.md)、[CPU 调度](./07-system/cpu/sched/README.md)、[x86 Cache](./07-system/cpu/x86/cache/README.md)、[内存系统](./07-system/memory/README.md)、[网络内核旁路](./07-system/net/kernel-bypass/README.md)、[io_uring](./07-system/os/io_uring/README.md)、[Cache Coherent](./07-system/cache/coherent/README.md)
- Profiling：[总览](./09-profile/README.md)、[CUDA 性能分析](./09-profile/cuda/README.md)、[GFlops 计算](./09-profile/cuda/theory.md)、[Benchmark](./09-profile/cuda/benchmark.md)、[Roofline](./09-profile/cuda/roofline.md)、[Warp Stall](./09-profile/cuda/stall.md)、[Kernel 延迟](./09-profile/latency/README.md)、[优化](./09-profile/improve/README.md)、[调试](./09-profile/debug/README.md)

---

## 当前重点

近期新增和重点维护方向：

- CUDA VMM、Pin Memory、Hopper Pipeline、Blackwell 架构
- Triton Matmul、FlashAttention、Kernel Fusion
- Chinchilla Scaling Law、训练算力与数据/参数配比
- PyTorch compile/custom ops/memory/linear 源码链路
- vLLM 架构、并行策略、显存管理、Sleep Mode
- 多模态 DiT/LDM/ADM、DiT Cache、Text2X
- FSDP/HSDP/ZeRO、CP 序列并行（Megatron-SP / Ring Attention / Ulysses）、跨卡同步机制（dist/P2P/IPC）、NCCL 与网络栈专题
- CPU 调度与绑核、x86 Cache 层次、内核旁路网络、epoll/io_uring
- Pre-Training / Post-Training（SFT/RLHF/DPO）训练流程
- GPU GFlops 计算、Roofline 性能建模、Warp Stall 硬件诊断、ILP 指令级并行
- 全局内存合并访问：transaction 模型、SoA vs AoS、GEMV 行主序/列主序
- 共享内存 Bank Conflict vs 全局内存 Uncoalesced 对比

待补主题见 [TODO.md](./TODO.md)。

---

## 维护约定

- 根 README 保持为高层入口，不追求列出每个叶子目录。
- 每个主题目录优先维护自己的 README，根 README 只链接稳定入口。
- 新增目录时尽量保持路径命名一致，避免大小写混用和拼写漂移。
- 示例代码、实验日志和图表应放在对应主题目录下，README 只保留结论、关键路径和复现实验入口。

最后更新：2026-06-20
