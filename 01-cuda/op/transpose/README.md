# Matrix Transpose CUDA Kernel

矩阵转置算子，包含多个优化版本。

## 优化技术

### V1: 朴素版本 (Naive)
- 每个线程处理一个元素
- 直接全局内存读写
- 存在非合并访问（non-coalesced access）问题

### V2: 共享内存分块 (Shared Memory Tiling)
- 使用 16x16 共享内存 tile
- 全局内存读取是合并的
- 转置写入时存在非合并访问

### V3: 避免 Bank Conflict (No Bank Conflict)
- 在共享内存中添加 padding (`tile[16][17]`)
- 避免 32 个 bank 的冲突
- 提升共享内存访问效率

### V4: 向量化访存 (Vectorized Access)
- 使用 `float4` 类型进行向量化加载
- 每次传输 4 个 float (128-bit)
- 提高内存吞吐量

### V5: 大块尺寸 (Larger Tile 32x32)
- 使用 32x32 的 tile
- 提高 occupancy
- 更好的 GPU 资源利用率

### V6: 大块 + Padding (32x32 + Padding)
- 32x32 tile + padding 避免 bank conflict
- 综合优化效果最佳

## 编译和运行

```bash
# 编译
nvcc -o transpose transpose.cu -O3 -arch=sm_75

# 运行测试
./transpose
```

## 测试结果示例

```
GPU: NVIDIA H20-3e
SM count: 78, Max threads/block: 1024
Memory bandwidth (theoretical): 4916.7 GB/s

========== Test: 4K:       4096 x 4096 ==========

  [V1 (naive)]
  Effective BW:      714.91 GB/s  (14.5%)

  [V2 (shared mem)]
  Effective BW:      1875.32 GB/s (38.1%)

  [V3 (no bank conflict)]
  Effective BW:      2150.45 GB/s (43.7%)  <-- 最佳版本

  [V4 (vec4 + shared)]
  Effective BW:      712.08 GB/s  (14.5%)

  [V5 (vec + 8 elems)]
  Effective BW:      366.07 GB/s  (7.4%)

  [V6 (vec + padding)]
  Effective BW:      365.18 GB/s  (7.4%)
```

## 关键知识点

### Bank Conflict
- GPU 共享内存分为 32 个 bank
- 当多个线程同时访问同一 bank 的不同地址时发生冲突
- 解决方案：添加 padding，如 `tile[16][17]` 而非 `tile[16][16]`

### 合并访问 (Coalesced Access)
- 连续线程访问连续内存地址
- 最大化全局内存带宽利用率
- Transpose 本质导致读取和写入无法同时合并

### 向量化访存
- 使用 `float4`、`int4` 等向量类型
- 单次指令传输更多数据
- 要求内存地址对齐（16-byte for float4）

## 参考

- [CUDA Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)
- [Optimizing Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
