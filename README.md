# AI-infra-LearningNote

## 1. CUDA

### 1.1 基础原语

[warp level](./cuda/primitives/warp/README.md)


### 1.2 规约操作

[reduce](./cuda/reduce/README.md)


### 1.3 向量化

[vectorize](./cuda/vectorize/)

### Hopper 架构特性

- [分布式共享内存](./cuda/hopper/DistributedSM/README.md)
- 

## 2. Trition

- [基础语法](./Triton/basic/README.md)
- [硬件信息](./Triton/hardware/README.md)
- [性能测试]()
- [随机数](./Triton/random/README.md)

## 3. 大模型

- [Attention](./LLMArch/Attention/README.md)
    - scaled Dot attention
    - MHA
    - MQA
    - GQA
- [MOE](./LLMArch/MoE/README.md)
    - basic MoE
    - sparse MoE

- [模型中间表示](./IR/README.md)
    - [PNNX](./IR/PNNX/README.md)
    - [ONNX](./IR/ONNX/README.md)
    - 可视化

- [模型保存](./model/save_load/README.md)

## 4. 训推优化

### 4.1 推理框架

- 推理框架
    - [TensoRT](./interferce/TensorRT/README.md)
    
### 4.1 量化

- 量化方法
    - [线性量化](./quant/linearQuant/README.md)
    - 非线性量化
    - 二值量化
    - [kmeans量化](./quant/kmeans/README.md)
    - [QAT](./quant/QAT/README.md)

### 4.2 模型剪枝

- [Fine-grain Pruning](./inference/prune/fine-grain/README.md)
- [channel-based Pruning](./inference/prune/channel-based/README.md)


### 4.3 Fine-Tuning

- [Supervised Fine-tuning](./train/Fine-tuning/SFT/README.md)
- [Reinforcement Learning from Human Feedback](./train/Fine-tuning/RLHF/README.md)
- [Parameter Efficient Fine-Tuning](./train/Fine-tuning/DPO/README.md)

### 4.4 模型学习率

[Learning Rate Schedules](./model/learningRT/README.md)
## 5 NAS

[OFA网络](./NAS/README.md)

## 6. LLM Benchmark

### 6.1 大模型问答评估

### 6.2 推理性能评估

推理性能指标一般为每秒输出的token数目

https://zhuanlan.zhihu.com/p/665170554
