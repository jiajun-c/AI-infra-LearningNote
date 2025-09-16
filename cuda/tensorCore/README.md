# TensorCore 加速

TensorCore 是英伟达GPU中的矩阵运算单元，我们可以通过mma指令来使用TensorCore 运算单元。mma指令是以warp为执行单位，

## 支持的数据type/shape


[具体文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m8n8k4#warp-level-matrix-shape)



## 数据布局

在mma运算指令中，需要以warp为单位进行执行，warp中的每个线程计算部分数据，然后将其累加到输出的位置，例如对于16x8x16 的half类型矩阵乘法。需要用到32个线程，对于矩阵A，将其分为四部分，每部分的开始两个元素，被放入到mma输入中。对于矩阵B，则是将一列上的四个元素放入到mma的输入中。最终的输出位置C的前半部分和后半部分。

```cpp
__global__ void mma_fp16_acc_fp32(float *out) {
  float c[4] = {0., 0., 0., 0.};
  float d[4] = {0., 0., 0., 0.};
  half a[8] = {1., 2., 1., 2., 1., 2., 1., 2.};
  half b[4] = {1., 2., 3., 4.};
  unsigned const *rA = reinterpret_cast<unsigned const *>(&a);
  unsigned const *rB = reinterpret_cast<unsigned const *>(&b);
  float const *rC = reinterpret_cast<float const *>(&c);
  float *rD = reinterpret_cast<float *>(&d);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(rD[0]), "=f"(rD[1]), "=f"(rD[2]), "=f"(rD[3])
      : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]), "r"(rB[0]), "r"(rB[1]),
        "f"(rC[0]), "f"(rC[1]), "f"(rC[2]), "f"(rC[3]));
  printf("%f\n", rD[0]);
  memcpy(out + threadIdx.x * 2, rD, 8);
  memcpy(out + 8 * 8 + threadIdx.x * 2, rD + 2, 8);
}
```
