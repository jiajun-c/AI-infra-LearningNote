# 语言与 Kernel DSL

先掌握宿主语言，再进入 GPU 编程 DSL 和库。各路线可以独立阅读。

1. [C++](./cpp/README.md)：类型、引用、模板、内存与并发。
2. [Python](./python/iter/README.md)：迭代器、生成器、异步与对象模型。
3. [Triton](./Triton/basic/README.md)：基础 → [矩阵乘法](./Triton/matmul/README.md) → [FlashAttention](./Triton/flashAttention/README.md)。
4. [CuTe DSL](./cuteDSL/layout/README.md)：Layout → [mbarrier](./cuteDSL/mbarrier/README.md) → [Pipeline](./cuteDSL/pipeline/README.md) → [TMA](./cuteDSL/tma/README.md)。
5. [CUTLASS 入门](./cutlass/start/)：将布局与流水线概念用于算子实现。

相关入口：[CUDA 与 GPU](../01-cuda/README.md)。
