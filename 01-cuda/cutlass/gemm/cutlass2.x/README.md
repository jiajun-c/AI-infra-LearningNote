# cutlass 2.x 矩阵乘法

cutlass2.x完全通过线性内存的视角去看待矩阵，其接口形式类似我们的cublas，如下所示是cutlass的一个函数签名，如果要去适配新的硬件特性较为困难

```cpp
// 2.x 的 Arguments 构造：简单、直白、像 cuBLAS
CutlassSgemmNN::Arguments args(
    {M, N, K},          // 1. 传总大小
    {d_A, lda},         // 2. 传 A 的指针和一维跨度 (lda)
    {d_B, ldb},         // 3. 传 B 的指针和一维跨度 (ldb)
    {d_C, ldc},         // 4. 传 C 的指针和一维跨度 (ldc)
    {d_D, ldd},         // 5. 传 D 的指针和一维跨度 (ldd)
    {alpha, beta}       // 6. 传缩放系数
);
```
在cutlass2.x中需要通过迭代器的形式去遍历不同的区域

性能上接近cublas，如下所示测试的性能结果

```cpp
GPU: NVIDIA H200
Benchmarking SGEMM (M=4096, N=4096, K=4096)
Warmup=10, Iterations=100

=============================================================
  SGEMM Performance Comparison (M=4096 N=4096 K=4096)
=============================================================
  CUTLASS 2.x    : 3.8300 ms  |  35.88 TFLOPS
  cuBLAS         : 2.6676 ms  |  51.52 TFLOPS
-------------------------------------------------------------
  Speedup (CUTLASS / cuBLAS) : 0.70x
=============================================================
```