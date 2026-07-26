# 算子

CUDA 高性能算子实现。

## 算子列表

| 算子 | 目录 | 关键优化 |
|------|------|---------|
| Transpose | [transpose/](transpose/README.md) | SMEM bank conflict 消除，vectorized load/store |
| Softmax | [softmax/](softmax/) | online softmax，warp reduce |
| Reduce | [reduce/](reduce/README.md) | warp shuffle reduce，block reduce |
| Element-wise | [element_wise/](element_wise/README.md) | 向量化访存，vectorize 模板 |
| GEMM | [gemm/](gemm/) | — |
| TopK | [topk/](topk/) | — |
| Block All-Reduce | [block_all_reduce/](block_all_reduce/) | warp shuffle，跨 block 同步 |

## 性能优化要点

- **访存合并**：对齐、SoA vs AoS、128-bit vectorized load
- **Shared Memory**：bank conflict 消除，swizzle/padding
- **计算**：TensorCore MMA、WGMMA、warp shuffle reduce
- **延迟隐藏**：cp.async pipeline、TMA + mbarrier

## 相关目录

- [BLAS 算子](../blas/) — hgemm/hgemv/vmulSum
- [CUTLASS](../cutlass/) — CUTLASS/CuTe 封装
- [Warp 原语](../primitives/warp/) — shuffle/ballot/sync
