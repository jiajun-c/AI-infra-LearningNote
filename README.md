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

## 3. 大模型基础

- [Attention](./LLMArch/Attention/README.md)
    - scaled Dot attention
    - MHA
    - MQA
    - GQA

## 4. 训推优化

### 4.1 量化
量化方法 
- [线性量化](./quant/linearQuant/README.md)
- 非线性量化
- 二值量化

## 5. LLM Benchmark

### 5.1 大模型问答评估

### 5.2 推理性能评估

推理性能指标一般为每秒输出的token数目

https://zhuanlan.zhihu.com/p/665170554
