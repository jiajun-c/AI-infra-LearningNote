# CUDA 与 GPU

从硬件执行模型进入内存、同步与算子实现；架构特性和库实现作为进阶阅读。

1. [硬件概览](./hardware/README.md) → [SM](./sm/README.md) → [Warp 原语](./primitives/warp/README.md)。
2. [内存](./memory/README.md) → [Stream](./stream/README.md) → [同步](./sync/README.md) → [Kernel 启动](./launch/README.md)。
3. [并行算法](./algorithm/README.md) → [基础算子](./op/) → [Tensor Core](./tensorCore/README.md) → [CUTLASS GEMM](./cutlass/gemm/cutlass3.x/README.md)。

架构专题：[Ampere](./ampere/)、[Hopper](./hopper/)、[Blackwell](./blackwell/README.md)。底层接口：[Driver API](./driver/README.md)、[PTX](./PTX/)、[JIT](./jit/README.md)。

相关入口：[语言与 Kernel DSL](../02-lang/README.md)、[CUDA 性能分析](../09-profile/cuda/README.md)。
