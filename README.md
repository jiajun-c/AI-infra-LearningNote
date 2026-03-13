# AI-infra-LearningNote

AI 基础设施学习笔记 - 涵盖 CUDA 编程、大模型架构、训练推理优化、通信库等核心主题

---

## 目录结构

```
AI-infra-LearningNote/
├── cuda/              # CUDA 编程与 GPU 架构
├── lang/              # 编程语言 (C++/Python/Triton)
├── LLMArch/           # 大模型架构组件
├── parallel/          # 并行训练策略
├── inference/         # 推理优化技术
├── quant/             # 量化方法
├── comm/              # 通信库与集合通信原语
├── framework/         # 深度学习框架
├── agent/             # AI Agent 相关
├── tools/             # 开发工具
├── system/            # 操作系统相关
├── xpu/               # XPU 架构
└── ...
```

---

## 1. CUDA 编程

### 1.1 基础
- [CUDA 架构演进](./cuda/arch/README.md) - Tesla 到 Blackwell 架构
- [启动配置](./cuda/launch/README.md)
- [Stream 管理](./cuda/stream/README.md)
- [合作组 (Cooperative Groups)](./cuda/cg/README.md)

### 1.2 核心原语
- [Warp 原语](./cuda/primitives/warp/README.md) - shfl, ballot, any, all 等
- [Reduce 优化](./cuda/reduce/README.md) - Warp/Block 级别 ReduceSum
- [Vectorize 访问](./cuda/vectorize/README.md)

### 1.3 TensorCore 编程
- [TensorCore 指令](./cuda/tensorCore/README.md) - mma 指令、数据布局、访存优化
- [PTX 内联汇编](./cuda/PTX/README.md)
  - [内联 PTX](./cuda/PTX/inline/README.md)

### 1.4 内存层次与优化
- [Bank 冲突优化](./cuda/memory/bank/README.md)
- [Cache 优化](./cuda/memory/cache/README.md)
- [地址空间判定](./cuda/memory/predicate/README.md)
- [内存空间转换](./cuda/memory/convert/README.md)

### 1.5 异步操作
- [Pipeline 机制](./cuda/sync/pipe/README.md)
- [Stream 同步](./cuda/sync/stream/README.md)
- [异步内存操作](./cuda/sync/mem/README.md)
- [块内同步](./cuda/sync/inner/README.md)

### 1.6 CUDA 算子实现
- **BLAS 算子**
  - [HGEMV](./cuda/blas/hgemv/README.md)
  - HGEMM ⚠️ TODO
- **逐元素算子**
  - [Element-wise](./cuda/op/element_wise/README.md)

### 1.7 Hopper 架构特性
- [分布式共享内存 (DSMEM)](./cuda/hopper/DistributedSM/README.md)
  - [DSMEM 算子优化](./cuda/dsmem/README.md)
- [TMA (Tensor Memory Accelerator)](./cuda/hopper/TMA/README.md)
- [Cluster 调度](./cuda/cutlass/cluster/)

### 1.8 CCCL 库
- [Thrust](./cuda/cccl/thrust/README.md)
- [cuSPARSE](./cuda/cccl/cusparse/README.md)

---

## 2. CUTLASS & CuTe

### 2.1 CuTe 基础
- [Layout 布局](./cuda/cutlass/cute/layout/layout.md)
- [Tensor 操作](./cuda/cutlass/cute/tensor/tensor.md)
- [多维度分块](./cuda/cutlass/cute/multidimTile/README.md)

### 2.2 GEMM 实现
- [高层 CuTe GEMM](./cuda/cutlass/gemm/cuteHigh/README.md)
- [Device 层 GEMM](./cuda/cutlass/gemm/device/README.md)

### 2.3 其他主题
- [Copy 机制](./cuda/cutlass/copy/README.md)
- [转置优化](./cuda/cutlass/trans/)
- [MMA 操作](./cuda/cutlass/cute/mma/)

---

## 3. 编程语言

### 3.1 C++
- **基础**
  - [类型系统](./lang/cpp/type/README.md)
  - [命名空间](./lang/cpp/namespace/README.md)
  - [类型转换](./lang/cpp/cast/README.md)
  - [自动类型推导](./lang/cpp/auto/README.md)
  - [Lambda 表达式](./lang/cpp/lamda/)

- **内存管理**
  - [内存管理](./lang/cpp/memory/README.md) - new/malloc/自定义内存管理器

- **面向对象**
  - [类继承](./lang/cpp/class/inherit/README.md)
  - [虚函数](./lang/cpp/class/virtual/README.md)
  - [显式构造函数](./lang/cpp/class/explicit/)
  - [模板](./lang/cpp/template/README.md)

- **元编程**
  - [元编程](./lang/cpp/metaprogam/README.md)

- **STL**
  - [vector](./lang/cpp/stl/vector/README.md)
  - [map](./lang/cpp/stl/map/README.md)
  - [unordered_map](./lang/cpp/stl/unordered_map/README.md)
  - [bitsets](./lang/cpp/stl/bitsets/README.md)

- **运算符**
  - [运算符重载](./lang/cpp/operator/README.md)

- **异步**
  - [future](./lang/cpp/async/future/README.md)

- **GCC 扩展**
  - [内建函数](./lang/cpp/gcc/builtin/)

### 3.2 Python
- [数据类型](./lang/python/type/README.md)
- [类系统](./lang/python/class/README.md)
  - [抽象基类](./lang/python/class/abc/README.md)
- [全局变量](./lang/python/global/)

### 3.3 Triton
- [基础语法](./lang/Triton/basic/README.md)
- [硬件信息](./lang/Triton/hardware/README.md)
- [性能测试](./lang/Triton/benchmark/README.md)
- [随机数生成](./lang/Triton/random/README.md)
- [矩阵乘法](./lang/Triton/matmul/)
- [FlashAttention](./lang/Triton/FlashAttention/)
- [LayerNorm](./lang/Triton/layernorm/)
- [Softmax](./lang/Triton/softmax/)
- [Autotune](./lang/Triton/autotune/)

### 3.4 CUTLASS
- [入门指南](./lang/cutlass/start/)

---

## 4. 大模型架构

### 4.1 整体流程
- [模型数据流](./LLMArch/flow/README.md)

### 4.2 Tokenization
- [BPE 分词](./tokenizer/BPE/README.md)

### 4.3 位置编码
- [绝对位置编码](./position_encode/absolute/README.md)
- [相对位置编码](./position_encode/relative/README.md)

### 4.4 Attention 机制
- [Attention 综述](./LLMArch/Attention/README.md)
  - Scaled Dot-product Attention
  - Multi-Head Attention (MHA)
  - Multi-Query Attention (MQA)
  - Grouped-Query Attention (GQA)
  - Softmax 优化
- [FlashAttention V1](./LLMArch/Attention/FlashAttention/README.md)
- [FlashAttention V2](./LLMArch/Attention/flashAttentionv2/README.md)
- [Ring Attention](./LLMArch/Attention/ring-attention/README.md)
- [Block-wise Attention](./LLMArch/Attention/blockWiseAttention/)
- [Unpad Attention](./LLMArch/Attention/unpad/)
- [Padding 处理](./LLMArch/Attention/pad.md)

### 4.5 MoE (Mixture of Experts)
- [MoE 基础](./LLMArch/MoE/README.md)
  - Basic MoE
  - Sparse MoE

### 4.6 线性层
- [线性层](./LLMArch/Linear/README.md)
  - LM Head
  - FFN/MLP

### 4.7 归一化
- [Norm 层](./LLMArch/Norm/README.md)

### 4.8 模型中间表示 (IR)
- [IR 基础](./IR/README.md)
- [PNNX](./IR/PNNX/README.md)
- [ONNX](./IR/ONNX/README.md)

### 4.9 模型保存与加载
- [模型保存/加载](./model/save_load/README.md)

### 4.10 深度学习基础
- [RNN](./dl/RNN/README.md)
- [LSTM](./dl/LSTM/README.md)
- [Word2Vec](./dl/word2vec/README.md)

---

## 5. 并行训练

### 5.1 数据并行
- [数据集处理](./parallel/dataset/README.md)
- [数据并行 (DP)](./parallel/DP/README.md)
- [分布式数据并行 (DDP)](./parallel/DDP/README.md)

### 5.2 模型并行
- [张量并行 (TP)](./parallel/TP/README.md)
  - 行并行/列并行
  - 序列并行
  - 并行损失函数
- [流水线并行 (PP)](./parallel/PP/README.md)

### 5.3 混合并行
- [混合并行策略](./parallel/HybirdParallel/README.md)
- [Expert Parallel (EP)](./parallel/EP/README.md)

---

## 6. 推理优化

### 6.1 推理框架
- [TensorRT](./inference/TensorRT/README.md)

### 6.2 批处理策略
- [批处理技术](./inference/batch/README.md)
  - 静态批处理
  - 动态批处理
  - 连续批处理 (Continuous Batching)
- [Prefix Cache](./inference/prefix_cache/README.md)
- [Chunked Prefill](./inference/chunkPrefill/README.md)

### 6.3 KV Cache 优化
- [KV Cache](./inference/kvcache/README.md)

### 6.4 模型剪枝
- [细粒度剪枝](./inference/prune/fine-grain/README.md)
- [通道剪枝](./inference/prune/channel-based/README.md)

---

## 7. 量化技术

### 7.1 基础量化
- [线性量化](./quant/linearQuant/README.md)
  - [对称量化](./quant/linearQuant/Symmetric/)
  - [非对称量化](./quant/linearQuant/Asymmetric/)
- [非线性量化](./quant/non-linear/) ⚠️ TODO
- [二值量化](./quant/binary/) ⚠️ TODO

### 7.2 量化方法
- [k-means 量化](./quant/kmeans/README.md)
- [QAT (量化感知训练)](./quant/QAT/README.md)
- [AWQ](./quant/AWQ/README.md)
- [SmoothQuant](./quant/smooth/README.md)
- [WNAM](./quant/WNAM/README.md)

---

## 8. 模型通信

### 8.1 通信后端
- [Gloo](./comm/backend/gloo/README.md)
- NCCL (see CCL)

### 8.2 集合通信原语
- [集合通信原语](./comm/collective/README.md)
  - All-Gather
  - Reduce-Scatter
  - All-Reduce
  - Broadcast
  - Send/Recv

### 8.3 通信库
- **NCCL**
  - [NCCL 基础](./comm/CCL/NCCL/README.md)
  - [配置选项](./comm/CCL/NCCL/config/README.md)
  - [图接口](./comm/CCL/NCCL/graph/README.md)
  - [Buffer 管理](./comm/CCL/NCCL/buffer/)
  - [Zero-CTA](./comm/CCL/NCCL/zero-CTA/)

---

## 9. 训练与微调

### 9.1 微调技术
- [监督微调 (SFT)](./train/Fine-tuning/SFT/README.md)
- [人类反馈强化学习 (RLHF)](./train/Fine-tuning/RLHF/README.md)
- [直接偏好优化 (DPO)](./train/Fine-tuning/DPO/README.md)

### 9.2 显存优化
- [梯度检查点](./train/LowMem/checkpoint/README.md)

### 9.3 学习率调度
- [学习率调度器](./model/learningRT/README.md)

---

## 10. 深度学习框架

### 10.1 PyTorch
- Tensor 操作
- [计算图](./framework/pytorch/graph/README.md)
- [分布式训练](./framework/pytorch/dist/README.md)
- [梯度机制](./framework/pytorch/grad/)
- [优化器](./framework/pytorch/optimizer/)
- [装饰器](./framework/pytorch/decorator/)

### 10.2 DeepSpeed
- [DeepSpeed 基础](./framework/deepspeed/README.md) ⚠️ TODO

---

## 11. Agent 与向量检索

### 11.1 Agent 框架
- [LangChain](./agent/langchain/README.md)
- [推理引擎](./agent/infer/README.md)

### 11.2 向量数据库
- [向量数据库基础](./agent/vectorDB/README.md)
- [Faiss](./agent/vectorDB/faiss/README.md)

---

## 12. 性能分析与调试

### 12.1 调试工具
- [调试基础](./debug/README.md)
- GDB 调试

### 12.2 性能分析
- [CUDA 性能分析](./profileAndHand/cuda/README.md)
  - [示例：GEMM](./profileAndHand/cuda/example/gemm/)
  - [示例：矩阵转置](./profileAndHand/cuda/example/matTrans/)
  - [示例：Warp Reduce](./profileAndHand/cuda/example/warpReduce/)
- [性能优化方法](./profileAndHand/improve/README.md)
- [Thop 工具](./profileAndHand/thop/)
- [困惑度分析](./profileAndHand/perplexity/)

### 12.3 日志系统
- Rank 日志分析

---

## 13. 操作系统基础

### 13.1 内存系统
- [页表](./system/memory/pagetable.md)
- [TLB](./system/memory/tlb.md)
- [Cache 一致性](./system/cache/coherent/)

### 13.2 进程与线程
- [进程/线程/协程](./system/process/README.md)

---

## 14. XPU 架构

### 14.1 GPU
- [GPU 架构](./xpu/gpu/README.md)

### 14.2 其他加速器
- [CPU (鲲鹏)](./xpu/cpu/kunpeng.md)
- [NPU](./xpu/npu/README.md)

---

## 15. 多模态

- [ViT](./multimodal/vit/README.md)
- [CLIP](./multimodal/clip/README.md)

---

## 16. 强化学习

- [RL 基础](./RL/README.md)
  - [表格型方法](./RL/tabular/)
  - [网络型方法](./RL/network/)

---

## 17. 知识蒸馏 (KD)

- 数据集与示例

---

## 18. 神经架构搜索 (NAS)

- [OFA 网络](./NAS/README.md)

---

## 19. 实用工具与第三方库

### 19.1 项目管理
- [Python 项目管理](./tools/pyproject/README.md)

### 19.2 第三方库
- [Einops](./third_party/einops/README.md) - 灵活的数据操作库

### 19.3 胶水层
- [Torch 绑定](./glue/torch/README.md)
  - [pybind11](./glue/torch/pybind/)
  - [nanobind](./glue/torch/nanobind/)

---

## 20. 编译器

### 20.1 TVM
- [TVM 基础](./compiler/tvm/README.md)
- [调优日志](./compiler/tvm/tuning_logs/)

---

## 21. 模型基准测试

### 21.1 推理性能
- [推理基准测试](./LLMBench/InferBench/README.md)

### 21.2 评估指标
- [语言能力评估](./LLMBench/metrics/README.md)
- [性能指标](./profileAndHand/metrics/)

---

## 22. 矩阵计算单元

- [ARM SME](./matrixUnit/arm_sme/)

---

## 23. 日志与调试

- Rank 日志分析

---

## 待办事项清单

### 文档完善（剩余）
- [ ] `cuda/blas/gemm/` - 添加 GEMM 实现文档
- [ ] `framework/deepspeed/README.md` - 完善 DeepSpeed 文档
- [ ] `quant/non-linear/` - 添加非线性量化文档
- [ ] `quant/binary/` - 添加二值量化文档

### 已完成
- [x] `cuda/launch/README.md` - 添加启动配置文档
- [x] `cuda/memory/cache/README.md` - 添加 Cache 优化文档
- [x] `inference/vllm/README.md` - 添加 vLLM 架构文档
- [x] `lang/Triton/benchmark/README.md` - 添加性能测试方法
- [x] `lang/cpp/class/inherit/README.md` - 添加继承文档
- [x] `position_encode/relative/README.md` - 添加相对位置编码文档
- [x] `xpu/README.md` - 添加 XPU 概述
- [x] `LLMArch/Attention/README.md` - 完善 MQA 部分
- [x] `parallel/TP/README.md` - 完善并行损失函数计算
- [x] `comm/CCL/NCCL/README.md` - 完善 AllGather 和点对点通信
- [x] `framework/pytorch/README.md` - 添加 PyTorch 概述
- [x] `model/save_load/README.md` - 添加模型保存/加载文档
- [x] `parallel/EP/` - 添加 EP 并行文档
- [x] 修正 `comm/collectivate/` -> `comm/collective/` (拼写错误)

---

## 参考资源

- [NVIDIA CUDA 文档](https://docs.nvidia.com/cuda/)
- [PyTorch 文档](https://pytorch.org/docs/)
- [Hugging Face](https://huggingface.co/docs)

---

最后更新：2026-03-06
