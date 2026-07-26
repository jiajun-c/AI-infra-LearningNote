# BLAS 算子

CUDA BLAS 级别的高性能算子实现。

## 算子列表

| 算子 | 目录 | 说明 |
|------|------|------|
| HGEMV | [hgemv/](hgemv/README.md) | Half-precision GEMV，含 thread/warp/block 多级实现和行主序/列主序对比 |
| HGEMM | [hgemm/](hgemm/) | Half-precision GEMM |
| vmulSum | [vmulSum/](vmulSum/) | 向量乘加求和 |

## 与 CUTLASS 的关系

- BLAS/ 是**手写 CUDA 实现**，侧重理解底层优化原理
- CUTLASS/ 是**框架封装**，侧重模板化复用

## 相关目录

- [算子总览](../op/README.md)
- [CUTLASS](../cutlass/)
- [TensorCore MMA](../tensorCore/README.md)
