# AI-infra-LearningNote

AI 基础设施学习笔记 - 涵盖 CUDA 编程、大模型架构、训练推理优化、通信库等核心主题

---

## 项目结构

```
AI-infra-LearningNote/
├── 01-cuda/           # CUDA 编程与 GPU 底层
├── 02-lang/           # 编程语言 (C++/Python/Triton)
├── 03-llm/            # 大模型（架构 + 训练 + 推理）
├── 04-comm/           # 通信库与集合通信原语
├── 05-framework/      # 深度学习框架
├── 06-agent/          # AI Agent 与向量检索
├── 07-system/         # 系统与硬件架构
├── 08-tools/          # 开发工具与第三方库
├── 09-profile/        # 性能分析与调试
├── cuda/              # CUTLASS/CuTe 实践代码
└── dao/               # 算子开发范式与任务划分
```

---

## 1. CUDA 编程 (01-cuda/)

### 架构与基础
- [Blackwell 架构](./01-cuda/blackwell/README.md) - UMMA 指令、双 SM 协同、TMA 到 TMem
- [启动配置](./01-cuda/launch/README.md)
- [Stream 管理](./01-cuda/stream/README.md)
- [合作组 (Cooperative Groups)](./01-cuda/cg/README.md)
- [JIT 编译 (NVRTC)](./01-cuda/jit/README.md) - 运行时特化、动态代码生成、架构自适应

### 核心原语
- [Warp 原语](./01-cuda/primitives/README.md) - shfl, ballot, any, all 等
- [Reduce 优化](./01-cuda/reduce/README.md) - Warp/Block 级别 ReduceSum
- [Vectorize 访问](./01-cuda/vectorize/README.md)

### TensorCore 编程
- [TensorCore 指令](./01-cuda/tensorCore/README.md) - mma 指令、数据布局、访存优化
- [PTX 内联汇编](./01-cuda/PTX/README.md)

### 内存层次与优化
- [Bank 冲突优化](./01-cuda/memory/bank/README.md)
- [Cache 优化](./01-cuda/memory/cache/README.md)
- [内存空间转换](./01-cuda/memory/convert/README.md)

### 异步操作
- [Pipeline 机制](./01-cuda/sync/pipe/README.md)
- [Stream 同步](./01-cuda/sync/stream/README.md)
- [异步内存操作](./01-cuda/sync/mem/README.md)
- [块内同步](./01-cuda/sync/inner/README.md)

### CUDA 算子实现
- **BLAS 算子**: [HGEMV](./01-cuda/blas/hgemv/README.md) | HGEMM ⚠️ TODO
- **逐元素算子**: [Element-wise](./01-cuda/op/element_wise/README.md) | [Vectorize Element-wise](./01-cuda/op/element_wise/vectorize/README.md)
- **Reduce 算子**: [ReduceMin](./01-cuda/op/reduce/) | [ReduceMax](./01-cuda/op/reduce/)
- **Softmax 算子**: [Softmax](./01-cuda/op/softmax/) - CUDA 与 Triton 实现

### Ampere 架构特性
- [cp.async 异步拷贝与流水线](./01-cuda/ampere/cpasync/README.md) - cp.async 指令、3-Stage Pipeline、性能对比

### Hopper 架构特性
- [分布式共享内存 (DSMEM)](./01-cuda/hopper/DistributedSM/README.md)
- [TMA (Tensor Memory Accelerator)](./01-cuda/hopper/TMA/README.md) - DMA 数据搬运、TMA 硬件引擎、调试示例
- [WGMMA (Warp Group MMA)](./01-cuda/hopper/wgmma/) - Warp Group 级矩阵乘加速指令
- [Pipeline 双缓冲](./01-cuda/hopper/pipe/README.md) - PipelineState、ClusterBarrier、生产者-消费者双缓冲
- [Cluster 调度](./01-cuda/cutlass/cluster/)

### CCCL 库
- [Thrust](./01-cuda/cccl/thrust/README.md)
- [cuSPARSE](./01-cuda/cccl/cusparse/README.md)

---

## 2. CUTLASS & CuTe (01-cuda/cutlass/)

### CuTe 基础
- [Layout 布局](./01-cuda/cutlass/cute/layout/layout.md)
- [Tensor 操作](./01-cuda/cutlass/cute/tensor/tensor.md)
- [多维度分块](./01-cuda/cutlass/cute/multidimTile/README.md)
- [_v/_t 后缀约定](./01-cuda/cutlass/cute/vt/README.md) - 编译期值提取

### GEMM 实现
- [高层 CuTe GEMM](./01-cuda/cutlass/gemm/cuteHigh/README.md)
- [Device 层 GEMM](./01-cuda/cutlass/gemm/device/README.md)
- [CUTLASS 3.x GEMM](./01-cuda/cutlass/gemm/cutlass3.x/README.md) - CuTe Layout 统一体系

### 其他主题
- [Copy 机制](./01-cuda/cutlass/copy/README.md)
- [转置优化](./01-cuda/cutlass/trans/)
- [MMA 操作](./01-cuda/cutlass/cute/mma/)

### CuTe 实践
- [Reduce 实现](./cuda/cutlass/cute/practice/) - 基于 CuTe 抽象的 Block 级 Reduce 实现

---

## 3. 编程语言 (02-lang/)

### C++
- **基础**: [类型系统](./02-lang/cpp/type/README.md) | [命名空间](./02-lang/cpp/namespace/README.md) | [类型转换](./02-lang/cpp/cast/README.md) | [自动类型推导](./02-lang/cpp/auto/README.md) | [Lambda 表达式](./02-lang/cpp/lamda/)
- **内存管理**: [内存管理](./02-lang/cpp/memory/README.md)
- **面向对象**: [类继承](./02-lang/cpp/class/inherit/README.md) | [虚函数](./02-lang/cpp/class/virtual/README.md) | [三/五法则](./02-lang/cpp/class/rules/README.md) | [模板](./02-lang/cpp/template/README.md)
- **元编程**: [元编程](./02-lang/cpp/metaprogam/README.md)
- **智能指针**: [shared_ptr / unique_ptr](./02-lang/cpp/point/)
- **STL**: [vector](./02-lang/cpp/stl/vector/README.md) | [map](./02-lang/cpp/stl/map/README.md) | [unordered_map](./02-lang/cpp/stl/unordered_map/README.md) | [bitsets](./02-lang/cpp/stl/bitsets/README.md)
- **运算符**: [运算符重载](./02-lang/cpp/operator/README.md)
- **异步**: [future](./02-lang/cpp/async/future/README.md)
- **GCC 扩展**: [内建函数](./02-lang/cpp/gcc/builtin/)

### Python
- [数据类型](./02-lang/python/type/README.md)
- [类系统](./02-lang/python/class/README.md) | [抽象基类](./02-lang/python/class/abc/README.md)
- [全局变量](./02-lang/python/global/)

### Triton
- [基础语法](./02-lang/Triton/basic/README.md)
- [硬件信息](./02-lang/Triton/hardware/README.md)
- [性能测试](./02-lang/Triton/benchmark/README.md)
- [随机数生成](./02-lang/Triton/random/README.md)
- [矩阵乘法](./02-lang/Triton/matmul/)
- [FlashAttention](./02-lang/Triton/FlashAttention/)
- [LayerNorm](./02-lang/Triton/layernorm/)
- [Softmax](./02-lang/Triton/softmax/)
- [Autotune](./02-lang/Triton/autotune/)
- [Kernel Fusion](./02-lang/Triton/fusion/permuteFusion/README.md) - Permute Fusion 四种方案对比

### CUTLASS
- [入门指南](./02-lang/cutlass/start/)

---

## 4. 大模型 (03-llm/)

### 4.1 模型架构 (arch/)
- **整体流程**: [模型数据流](./03-llm/arch/flow/README.md)
- **Tokenization**: [BPE 分词](./03-llm/arch/tokenizer/BPE/README.md)
- **位置编码**: [绝对位置编码](./03-llm/arch/position_encode/absolute/README.md) | [相对位置编码](./03-llm/arch/position_encode/relative/README.md)
- **Attention 机制**: [Attention 综述](./03-llm/arch/Attention/README.md)
  - [FlashAttention V1](./03-llm/arch/Attention/FlashAttention/README.md)
  - [FlashAttention V2](./03-llm/arch/Attention/flashAttentionv2/README.md)
  - [Ring Attention](./03-llm/arch/Attention/ring-attention/README.md)
  - [Block-wise Attention](./03-llm/arch/Attention/blockWiseAttention/)
  - [Unpad Attention](./03-llm/arch/Attention/unpad/)
- **MoE**: [MoE 基础](./03-llm/arch/MoE/README.md)
- **线性层**: [线性层](./03-llm/arch/Linear/README.md)
- **归一化**: [Norm 层](./03-llm/arch/Norm/README.md)
- **模型中间表示 (IR)**: [IR 基础](./03-llm/IR/README.md) | [PNNX](./03-llm/IR/PNNX/README.md) | [ONNX](./03-llm/IR/ONNX/README.md)
- **模型保存与加载**: [模型保存/加载](./03-llm/model/save_load/README.md)
- **权重形状分析**: [Weight Shape 分析](./03-llm/model/weightShape/) - LLM 线性层权重维度分析与 GEMM 模式识别
- **学习率调度**: [学习率调度器](./03-llm/model/learningRT/README.md)

### 4.2 深度学习基础 (foundation/)
- [RNN](./03-llm/foundation/dl/RNN/README.md)
- [LSTM](./03-llm/foundation/dl/LSTM/README.md)
- [Word2Vec](./03-llm/foundation/dl/word2vec/README.md)

### 4.3 并行训练 (train/)
- **数据并行**: [数据集处理](./03-llm/train/dataset/README.md) | [DP](./03-llm/train/DP/README.md) | [DDP](./03-llm/train/DDP/README.md)
- **模型并行**: [张量并行 (TP)](./03-llm/train/TP/README.md) | [流水线并行 (PP)](./03-llm/train/PP/README.md)
- **混合并行**: [混合并行策略](./03-llm/train/HybirdParallel/README.md) | [Expert Parallel (EP)](./03-llm/train/EP/README.md)
- **微调技术**: [SFT](./03-llm/train/finetuning/SFT/README.md) | [RLHF](./03-llm/train/finetuning/RLHF/README.md) | [DPO](./03-llm/train/finetuning/DPO/README.md)
- **显存优化**: [梯度检查点](./03-llm/train/LowMem/checkpoint/README.md)

### 4.4 推理优化 (inference/)
- **推理框架**: [TensorRT](./03-llm/inference/TensorRT/README.md)
- **批处理策略**: [批处理技术](./03-llm/inference/batch/README.md) | [Continuous Batching](./03-llm/inference/contiuousBatching/README.md) | [Chunked Prefill](./03-llm/inference/chunkPrefill/README.md)
- **KV Cache**: [KV Cache](./03-llm/inference/kvcache/README.md) | [Prefix Cache](./03-llm/inference/prefix_cache/README.md)
- **FlashDecode**: [FlashDecode](./03-llm/inference/flashDecode/README.md)
- **投机采样**: [Speculative](./03-llm/inference/speculative/README.md)
- **模型剪枝**: [细粒度剪枝](./03-llm/inference/prune/fine-grain/README.md) | [通道剪枝](./03-llm/inference/prune/channel-based/README.md)
- **量化技术**:
  - [线性量化](./03-llm/inference/quant/linearQuant/README.md) | [对称量化](./03-llm/inference/quant/linearQuant/Symmetric/) | [非对称量化](./03-llm/inference/quant/linearQuant/Asymmetric/)
  - [QAT](./03-llm/inference/quant/QAT/README.md) | [AWQ](./03-llm/inference/quant/AWQ/README.md) | [SmoothQuant](./03-llm/inference/quant/smooth/README.md) | [WNAM](./03-llm/inference/quant/WNAM/README.md)
  - [k-means 量化](./03-llm/inference/quant/kmeans/README.md)
  - 非线性量化 ⚠️ TODO | 二值量化 ⚠️ TODO

### 4.5 其他主题
- **多模态**: [ViT](./03-llm/multimodal/vit/README.md) | [CLIP](./03-llm/multimodal/clip/README.md)
- **强化学习**: [RL 基础](./03-llm/RL/README.md)
- **知识蒸馏 (KD)**
- **神经架构搜索 (NAS)**: [OFA 网络](./03-llm/NAS/README.md)

### 4.6 性能基准 (bench/)
- [推理基准测试](./03-llm/bench/InferBench/README.md)
- [评估指标](./03-llm/bench/metrics/README.md)

---

## 5. 模型通信 (04-comm/)

### 通信后端
- [Gloo](./04-comm/backend/gloo/README.md)
- [NCCL](./04-comm/CCL/NCCL/README.md) | [配置选项](./04-comm/CCL/NCCL/config/README.md) | [图接口](./04-comm/CCL/NCCL/graph/README.md)

### 集合通信原语
- [集合通信原语](./04-comm/collective/README.md) - All-Gather, Reduce-Scatter, All-Reduce, Broadcast, Send/Recv

---

## 6. 深度学习框架 (05-framework/)

### PyTorch
- [计算图](./05-framework/pytorch/graph/README.md)
- [分布式训练](./05-framework/pytorch/dist/README.md)
- [显存管理](./05-framework/pytorch/memory/model/README.md)
- [torch.compile 优化](./05-framework/pytorch/compile/README.md) - JIT 编译优化、Graph Break 分析
- [自定义 CUDA 算子](./05-framework/pytorch/custom_ops/README.md) - pybind vs torch.library 绑定方式与 CUDA Graph 兼容性

### DeepSpeed
- [DeepSpeed 基础](./05-framework/deepspeed/README.md) ⚠️ TODO

### vLLM
- [vLLM 框架](./05-framework/vllm/README.md)

---

## 7. AI Agent (06-agent/)

### Agent 框架
- [LangChain](./06-agent/langchain/README.md)
- [推理引擎](./06-agent/infer/README.md)

### 向量数据库
- [向量数据库基础](./06-agent/vectorDB/README.md)
- [Faiss](./06-agent/vectorDB/faiss/README.md)

---

## 8. 系统与硬件 (07-system/)

### 内存系统
- [页表](./07-system/memory/pagetable.md)
- [TLB](./07-system/memory/tlb.md)
- [Cache 一致性](./07-system/cache/coherent/)

### 进程与线程
- [进程/线程/协程](./07-system/process/README.md)

### 硬件架构
- [GPU 架构](./07-system/gpu/README.md)
- [CPU (鲲鹏)](./07-system/cpu/kunpeng.md)
- [NPU](./07-system/npu/README.md)
- [ARM SME](./07-system/matrixUnit/arm_sme/)

---

## 9. 工具与性能分析 (08-tools/ & 09-profile/)

### 开发工具 (08-tools/)
- [Python 项目管理](./08-tools/pyproject/README.md)
- [TVM 编译器](./08-tools/compiler/tvm/README.md)
- [Einops](./08-tools/third_party/einops/README.md)
- [Torch 绑定](./08-tools/glue/torch/README.md) | [pybind11](./08-tools/glue/torch/pybind/) | [nanobind](./08-tools/glue/torch/nanobind/)
- [Tuning](./08-tools/tuning/)

### 性能分析 (09-profile/)
- [CUDA 性能分析](./09-profile/cuda/README.md)
  - [示例：GEMM](./09-profile/cuda/example/gemm/)
  - [示例：矩阵转置](./09-profile/cuda/example/matTrans/)
  - [示例：Warp Reduce](./09-profile/cuda/example/warpReduce/)
- [性能优化方法](./09-profile/improve/README.md)
- [调试基础](./09-profile/debug/README.md)
- [Thop 工具](./09-profile/thop/)
- [困惑度分析](./09-profile/perplexity/)
- [日志分析](./09-profile/log/)

---

## 10. 算子开发范式 (dao/)

- [算子开发范式](./dao/README.md) - 算子开发思考
- [任务划分策略](./dao/partition/README.md) - 维度中心 vs 硬件中心 vs Split-K

---

## 待办事项

### 文档完善
- [ ] `01-cuda/blas/gemm/` - 添加 GEMM 实现文档
- [ ] `05-framework/deepspeed/` - 完善 DeepSpeed 文档
- [ ] `03-llm/inference/quant/non-linear/` - 添加非线性量化文档
- [ ] `03-llm/inference/quant/binary/` - 添加二值量化文档

### 已完成
- [x] 项目结构重构 - 统一目录命名与分类
- [x] `01-cuda/launch/` - 启动配置文档
- [x] `01-cuda/memory/cache/` - Cache 优化文档
- [x] `05-framework/vllm/` - vLLM 架构文档
- [x] `02-lang/Triton/benchmark/` - 性能测试方法
- [x] `03-llm/arch/Attention/` - MQA 部分完善
- [x] `03-llm/train/TP/` - 并行损失函数计算
- [x] `04-comm/CCL/NCCL/` - AllGather 和点对点通信
- [x] `03-llm/model/save_load/` - 模型保存/加载文档
- [x] `03-llm/train/EP/` - EP 并行文档
- [x] `01-cuda/blackwell/` - Blackwell 架构文档 (UMMA、双 SM 协同)
- [x] `01-cuda/jit/` - CUDA JIT 编译 (NVRTC) 文档
- [x] `01-cuda/hopper/pipe/` - Hopper 双缓冲流水线示例
- [x] `01-cuda/op/softmax/` - Softmax 算子实现
- [x] `01-cuda/op/reduce/` - Reduce 算子实现
- [x] `01-cuda/op/element_wise/vectorize/` - Vectorize Element-wise 算子
- [x] `02-lang/Triton/fusion/` - Triton Kernel Fusion (Permute Fusion)
- [x] `02-lang/cpp/class/rules/` - C++ 三/五法则
- [x] `02-lang/cpp/point/` - 智能指针 (shared_ptr/unique_ptr)
- [x] `05-framework/pytorch/compile/` - torch.compile 优化
- [x] `05-framework/pytorch/custom_ops/` - 自定义 CUDA 算子绑定
- [x] `01-cuda/cutlass/gemm/cutlass3.x/` - CUTLASS 3.x GEMM
- [x] `01-cuda/cutlass/cute/vt/` - CuTe _v/_t 后缀约定
- [x] `dao/` - 算子开发范式与任务划分

---

## 参考资源

- [NVIDIA CUDA 文档](https://docs.nvidia.com/cuda/)
- [PyTorch 文档](https://pytorch.org/docs/)
- [Hugging Face](https://huggingface.co/docs)

---

最后更新：2026-03-29
