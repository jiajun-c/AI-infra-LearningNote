# cp.async

在Ampere架构的机器下，进行矩阵运算的逻辑如下所示

Global Memory $\xrightarrow{\text{cp.async}}$ Shared Memory $\xrightarrow{\text{ldmatrix}}$ Registers $\xrightarrow{\text{mma.sync}}$ Tensor

接下来我们要介绍cp.async

## 背景：为什么需要 cp.async

在Ampere之前的机器的上，如果想把数据拷贝到share memory中时，数据需要在寄存器中进行中转

Global Memory $\rightarrow$ L2 Cache $\rightarrow$ Registers $\rightarrow$ Shared Memory

引入cp.async后，变为如下，直接跳过了寄存器，除了bypass 寄存器外，还有异步的优点，可以实现计算和通信重叠

Global Memory $\rightarrow$ L2 Cache $\rightarrow$ Shared Memory

## 指令语法

cp.async 指令声明如下所示

```shell
cp.async.cg.shared.global [%0], [%1], %2;
```

- %0: Shared Memory 的目标指针。
- %1: Global Memory 的源指针。
- %2: 要搬运的字节数（支持 4, 8, 16 字节）。为了极限带宽，工业界通常强制要求凑齐 16 字节（128-bit，即 float4 或 int4）对齐搬运。

下面这个写法则是表示等待在空中的组小于或者等于两组为止

```shell
asm volatile("cp.async.wait_group 2;\n" ::);
```

在程序的结束的时候保证其等于0组即可

## Demo：三种实现对比

本示例实现了三种 kernel，对同一任务（float4 向量逐元素乘 2）进行性能对比，展示 cp.async 配合流水线的收益。

### Kernel 1：Naive（传统路径，无流水线）

传统的 Global → Register → Shared 路径，数据经过寄存器中转，每次迭代同步加载一块数据到共享内存，再计算，再写回。**加载与计算完全串行**。

```cpp
for (int i = 0; i < tiles_per_block; ++i) {
    // Global → Register → Shared（传统路径，数据经寄存器中转）
    s_data[tid] = g_in[block_offset + i * tile_size + tid];
    __syncthreads();

    // 计算
    float4 val = s_data[tid];
    val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;
    g_out[block_offset + i * tile_size + tid] = val;
    __syncthreads();
}
```

### Kernel 2：cp.async 无流水线

使用 cp.async 绕过寄存器直达 Shared Memory，但每次加载后立即 `wait_group 0` 阻塞等待。**展示 cp.async 本身的特性，但没有流水线带来的计算/访存重叠**。

```cpp
for (int i = 0; i < tiles_per_block; ++i) {
    // cp.async: Global → Shared（绕过寄存器）
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(global_ptr));
    asm volatile("cp.async.commit_group;\n" ::);

    // 立即等待 —— 没有流水线，完全阻塞
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // 计算并写回...
}
```

### Kernel 3：cp.async + 3-Stage Pipeline（计算与访存重叠）

cp.async + 3 级流水线，使用 3 个共享内存 buffer 轮转。在等待第 i 块数据时，已经发起了第 i+2 块的预取，**实现计算与访存的深度重叠**。

流水线分为三个阶段：

```
┌─────────────────────────────────────────────────────────┐
│ Prologue: 预发起 2 块数据的异步拷贝，填满流水线         │
│   cp.async(tile 0) → commit                             │
│   cp.async(tile 1) → commit                             │
├─────────────────────────────────────────────────────────┤
│ Mainloop: 每次迭代同时做三件事                          │
│   ① 发起第 i+2 块的拷贝（prefetch）                    │
│   ② 等待第 i 块的数据到达（wait_group 2）              │
│   ③ 计算第 i 块并写回                                  │
├─────────────────────────────────────────────────────────┤
│ Epilogue: 处理最后 2 块数据，清空流水线                 │
│   wait_group 1 → 计算倒数第 2 块                       │
│   wait_group 0 → 计算最后 1 块                         │
└─────────────────────────────────────────────────────────┘
```

## 编译与运行

```shell
nvcc -O3 -arch=sm_80 demo_cpasync.cu -o demo_cpasync
./demo_cpasync
```

## 性能对比结果

测试配置：256 blocks x 64 tiles/block x 128 threads/block，数据量 32MB

```shell
============================================================
  cp.async Pipeline vs No-Pipeline 性能对比
============================================================
配置: 256 blocks × 64 tiles/block × 128 threads/block
数据量: 8388608 floats = 32.00 MB
------------------------------------------------------------

[Kernel 1] Naive (Global→Reg→Shared, 无流水线)
  正确性: PASS ✓
  平均耗时: 0.0308 ms

[Kernel 2] cp.async 无流水线 (每次加载后立即 wait)
  正确性: PASS ✓
  平均耗时: 0.0464 ms

[Kernel 3] cp.async + 3-Stage Pipeline (计算与访存重叠)
  正确性: PASS ✓
  平均耗时: 0.0255 ms
```

| Kernel | 方式 | 平均耗时 | 相对 Pipeline 的加速比 |
|--------|------|---------|----------------------|
| Kernel 1 (Naive) | Global→Reg→Shared, 串行 | 0.0308 ms | 0.83x |
| Kernel 2 (cp.async 无pipe) | cp.async 但立即等待 | 0.0464 ms | 0.55x |
| Kernel 3 (3-Stage Pipeline) | cp.async + 流水线重叠 | 0.0255 ms | **1.00x (最快)** |

### 结论

- **cp.async 单独使用（无流水线）反而比朴素版更慢**：虽然绕过了寄存器中转，但 `commit_group` / `wait_group` 指令本身有额外开销，且没有实现访存与计算的重叠
- **cp.async 的真正威力在于配合多级流水线**：通过 3 个 shared memory buffer 轮转，在计算当前数据块的同时预取后续数据块，实现了计算与访存的深度重叠，获得最优性能
- 流水线级数（本例为 3 级）需要权衡：级数越多，shared memory 占用越大，但能隐藏更长的访存延迟
