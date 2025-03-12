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

## 4. 训推优化

### 4.1 推理框架

- 推理框架
    - [TensoRT](./interferce/TensorRT/README.md)
    
### 4.1 量化
- 量化方法 
    - [线性量化](./quant/linearQuant/README.md)
    - 非线性量化
    - 二值量化


## 5. LLM Benchmark

### 5.1 大模型问答评估

### 5.2 推理性能评估

推理性能指标一般为每秒输出的token数目

https://zhuanlan.zhihu.com/p/665170554
