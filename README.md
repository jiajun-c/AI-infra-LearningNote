# AI-infra-LearningNote
## 1. CUDA

[CUDA架构/编译](./cuda/arch/README.md)

### 1.1 基础操作

- [warp原语](./cuda/primitives/warp/README.md)
- [reduce](./cuda/reduce/README.md)
- [vectorize](./cuda/vectorize/README.md)

### 1.2 TensorCore

[TensorCore](./cuda/tensorCore/README.md)

### CUDA 异步操作
- [cuda::pipeline](./cuda/sync/pipe/README.md)
- [stream](./cuda/sync/stream/README.md)
- [异步访存操作](./cuda/sync/mem/README.md)

### CUDA算子实现

- blas算子
    - [gemv](./cuda/blas/hgemv/README.md)
    - [gemm](./cuda/blas/gemm/README.md)

### 访存

- 访存优化
    - [bank 冲突优化](./cuda/memory/bank/README.md)
    - [cache 优化](./cuda/memory/cache/README.md)
- 地址空间
    - [地址空间判定函数](./cuda/memory/predicate/README.md)
    - [内存空间转换](./cuda/memory/convert/README.md)

### Hopper 架构特性

- [分布式共享内存](./cuda/hopper/DistributedSM/README.md)

- CCCL
    - [thrust](./cuda/cccl/thrust/README.md)
    - [cusparse](./cuda/cccl/cusparse/README.md)


## 2. 编程语言

### 2.1 C++

> 为了更好的写cuda :(

- [元编程](./lang/cpp/metaprogam/README.md)
- [内存管理](./lang/cpp/memory/README.md)
    - new malloc ...
    - 自定义内存管理器
- STL库
    - [bitsets](./lang/cpp/stl/bitsets/README.md)
    - [map](./lang/cpp/stl/map/README.md)
    - [vector](./lang/cpp/stl/vector/README.md)
    - [unordered_map](./lang/cpp/stl/unordered_map/README.md)

- [运算符](./lang/cpp/operator/README.md)
    - 运算符重载

- 异步操作
    - [future](./lang/cpp/async/future/README.md) 

- [命名空间](./lang/cpp/namespace/README.md)

### 2.2 Python

### 2.3 Trition

- [基础语法](./lang/Triton/basic/README.md)
- [硬件信息](./lang/Triton/hardware/README.md)
- [性能测试]()
- [随机数](./lang/Triton/random/README.md)

## 3. 大模型
- 编程框架
    - [PyTorch](./pytorch/README.md)
- 文本token化
    - [BPE](./tokenizer/BPE/README.md)
- 位置编码
    - [绝对位置编码](./position_encode/absolute/README.md)
    - [相对位置编码]()
- [Attention](./LLMArch/Attention/README.md)
    - scaled Dot attention
    - MHA
    - MQA
    - GQA
    - [softmax](./LLMArch/Attention/)
- [MOE](./LLMArch/MoE/README.md)
    - basic MoE
    - sparse MoE

- [模型中间表示](./IR/README.md)
    - [PNNX](./IR/PNNX/README.md)
    - [ONNX](./IR/ONNX/README.md)
    - 可视化

- [模型保存](./model/save_load/README.md)

- 模型并行
    - [并行数据集处理](./parallel/dataset/README.md)
    - [DP(数据并行)](./parallel/DP/README.md)
    - [DDP(分布式数据并行)](./parallel/DDP/README.md)
    - [TP(模型并行)](./parallel/TP/README.md)
    - [PP(流水并行)](./parallel/PP/README.md)
## 4. 训推优化

### 4.1 推理框架

- 推理框架
    - [TensoRT](./interferce/TensorRT/README.md)

- 训练框架
    - [deepspeed](./framework/deepspeed/README.md)
    
### 4.2 显存优化

- 量化方法
    - [线性量化](./quant/linearQuant/README.md)
    - 非线性量化
    - 二值量化
    - [kmeans量化](./quant/kmeans/README.md)
    - [QAT](./quant/QAT/README.md)
- 检查点机制
    - [梯度检查点](./train/LowMem/checkpoint/README.md)

### 4.3 模型剪枝

- [Fine-grain Pruning](./inference/prune/fine-grain/README.md)
- [channel-based Pruning](./inference/prune/channel-based/README.md)

### 4.4 Fine-Tuning

- [Supervised Fine-tuning](./train/Fine-tuning/SFT/README.md)
- [Reinforcement Learning from Human Feedback](./train/Fine-tuning/RLHF/README.md)
- [Parameter Efficient Fine-Tuning](./train/Fine-tuning/DPO/README.md)

### 4.5 模型学习率

[Learning Rate Schedules](./model/learningRT/README.md)

## 5. 模型通信

### 5.1 通信后端

- [gloo](./comm/backend/gloo/README.md)


### 5.2 通信原语

- [集合通信原语](./comm/collectivate/README.md)

### 5.3 通信库

- [NCCL](./comm/CCL/NCCL/README.md)

## 6. Agent

- [langChain](./agent/langchain/README.md)
- [推理](./agent/infer/README.md)
- [向量数据库](./agent/vectorDB/README.md)
  - [faiss](./agent/vectorDB/faiss/README.md)

## NAS

[OFA网络](./NAS/README.md)

## LLM Benchmark

### 大模型问答评估

### 推理性能评估

推理性能指标一般为每秒输出的token数目

https://zhuanlan.zhihu.com/p/665170554


## 实用第三方库

- [einops](./third_party/einops/README.md): 实用的数据操作库

## 性能优化

[分析程序](./profile/improve/README.md)


## 操作系统相关

- 存储
    - [页表](./system/memory/pagetable.md)
    - [TLB](./system/memory/tlb.md)
- 进程
    - []()