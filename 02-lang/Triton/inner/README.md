# Triton 原理

triton是一个block-level的GPU kernel的编译式DSL，

## 1. 核心范式改变

从SIMT这样一个thread centric转换为了block centric，一个`@triton.jit`函数就是一个program实例，写的操作是在整块的上做操作

## 2. 编译流水线

Triton限制是构建在MLIR上的多级lowering

Python AST -> Triton IR -> TritonGPU IR(ttgir) -> LLVM IR -> PTX -> cubin

- Layout inference/propagation
- 自动访存合并+向量化
- 共享内存管理+swizzle
- 软件流水，当他识别出 load -> matmul -> store 的循环的时候就会按照多级流水的形式对三步操作进行流水化，实现计算和搬运的overlap
