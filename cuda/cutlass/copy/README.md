# cutlass拷贝

cutlass中提供了`copy` 和 `copy_if`的接口用于进行数据的拷贝， 最普通的数据拷贝形式为`copy(src, dst)`，但是这样的拷贝实际上大概率会调用`ld.global.f32` 和 `st.global.f32`，对于我们来说其实际调用的异步/向量/标量是黑盒。

## 1. 基础数据拷贝

cutlass(CuTe)中最基础的数据拷贝方式是直接调用`copy(src, dst)`，它接受两个Tensor并逐元素地将源数据搬运到目标地址。

### 1.1 最简单的copy

最朴素的调用方式如下，直接构建两个Tensor然后调用`copy`：

```cpp
using namespace cute;

// 构建全局内存 Tensor
auto mIn  = make_tensor(make_gmem_ptr(d_in),  make_shape(N), make_stride(Int<1>{}));
auto mOut = make_tensor(make_gmem_ptr(d_out), make_shape(N), make_stride(Int<1>{}));

// 直接拷贝：编译器会根据类型自动选择指令
// 大概率生成 ld.global.f32 + st.global.f32（标量加载/存储）
copy(mIn, mOut);
```

这种方式的问题在于：**编译器自动决定拷贝指令的类型**（标量/向量/异步），对于使用者来说是黑盒的，无法精确控制指令行为。

### 1.2 Copy_Atom: 拷贝原子操作

`Copy_Atom`是CuTe中拷贝的最小粒度单元，它**显式指定了底层使用什么指令进行拷贝**，常见的`Copy_Atom`类型有：

| Copy_Atom 类型 | 说明 | 最低架构 |
|---|---|---|
| `UniversalCopy<T>` | 通用拷贝，T决定一次搬运的位宽（如`uint128_t`为128bit） | 任意 |
| `SM80_CP_ASYNC_CACHEALWAYS<T>` | cp.async异步拷贝（Global→Shared），始终经过L2 cache | SM80 (Ampere) |
| `SM80_CP_ASYNC_CACHEGLOBAL<T>` | cp.async异步拷贝，绕过L1 cache | SM80 (Ampere) |
| `SM90_TMA_LOAD` | TMA加载（Global→Shared），硬件DMA引擎 | SM90 (Hopper) |
| `SM90_TMA_STORE` | TMA存储（Shared→Global），硬件DMA引擎 | SM90 (Hopper) |

例如，通过指定`UniversalCopy<float>`，可以强制每次只搬运一个float（32bit标量）：

```cpp
// 标量拷贝：每次搬运1个float（32bit）
Copy_Atom<UniversalCopy<float>, float> scalar_atom;
```

而指定`UniversalCopy<uint128_t>`，则强制使用128bit向量指令，一次搬运4个float：

```cpp
// 向量拷贝：每次搬运128bit = 4个float
Copy_Atom<UniversalCopy<uint128_t>, float> vec_atom;
```

### 1.3 手动分块拷贝

在不使用`make_tiled_copy`的情况下，也可以手动进行分块拷贝。核心思路是：**将一维数据重塑为二维矩阵，一个维度表示线程编号，另一个维度表示每个线程搬运的元素数**，然后每个线程用自己的`threadIdx.x`去索引属于自己的那一部分数据：

```cpp
__global__ void cute_copy_kernel_manual(float const* d_in, float* d_out, int N) {
    using namespace cute;

    using BlockThreads = Int<128>; // 128个线程
    using VecElem      = Int<4>;   // 每个线程搬4个元素

    // 每个Block处理 128 * 4 = 512 个元素
    auto BLOCK_TILE_SIZE = size(BlockThreads{}) * size(VecElem{});
    int blk_idx = blockIdx.x;
    if (blk_idx * BLOCK_TILE_SIZE >= N) return;

    // 将一段连续内存重塑为 (128, 4) 的二维视图
    // Stride: (4, 1) -> 同一线程的4个元素在内存中连续
    auto block_shape  = make_shape(BlockThreads{}, VecElem{});
    auto block_stride = make_stride(VecElem{}, Int<1>{});

    auto gIn  = make_tensor(make_gmem_ptr(d_in  + blk_idx * BLOCK_TILE_SIZE),
                            make_layout(block_shape, block_stride));
    auto gOut = make_tensor(make_gmem_ptr(d_out + blk_idx * BLOCK_TILE_SIZE),
                            make_layout(block_shape, block_stride));

    // 每个线程取自己的那一行（4个连续元素）
    auto tIgIn  = gIn(threadIdx.x, _);   // shape: (4,)
    auto tOgOut = gOut(threadIdx.x, _);   // shape: (4,)

    // 调用copy，编译器根据类型推断指令
    copy(tIgIn, tOgOut);
}
```

内存布局示意图如下，`T0`-`T127`表示线程编号，每个线程负责搬运4个连续的float：

```
内存地址:  |  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | ... | 508 509 510 511 |
线程编号:  |     T0      |     T1      |     T2      | ... |      T127       |
           |<- 4 float ->|<- 4 float ->|<- 4 float ->|     |<-  4 float   -> |
```

这种方式虽然可以控制每个线程搬运的元素数量，但**无法显式指定底层拷贝指令**（如强制128bit向量化）。要精确控制指令行为，需要使用`make_tiled_copy`（见第2节）。

### 1.4 小结

基础拷贝的三种方式对比：

| 方式 | 是否可控制指令 | 是否可控制分块 | 适用场景 |
|---|---|---|---|
| `copy(src, dst)` | ❌ 编译器自动选择 | ❌ | 快速原型验证 |
| 手动重塑 + `copy` | ❌ 编译器自动选择 | ✅ | 需要自定义分块但不需精确控制指令 |
| `make_tiled_copy` + `Copy_Atom` | ✅ 显式指定 | ✅ | 生产级代码，需要精确控制性能 |

## 2. 详解make_tiled_copy

`make_tiled_copy` 是用于创建一个线程块级别的copyEngine其定义如下所示

```cpp
auto tiled_copy = make_tiled_copy(
    Copy_Atom,      // 参数1：底层拷贝原子指令
    ThreadLayout,   // 参数2：线程排布方式
    ValueLayout     // 参数3：单线程的数据排布方式
);
```

- 参数1是底层原子拷贝指令：`UniversalCopy<uint128_t>`表示一次向量化加载128个字节
- 参数二是线程的排布方式
- 参数三是单线程的数据排布方式：一个线程在不同维度上每次读取多少个数据


```cpp
    // Atom: UniversalCopy<uint128_t> -> 强制使用 128bit 向量指令 (一次搬4个float)
    // Thread Layout: 32x8 (列主序) -> 32行8列的线程阵列
    // Value Layout:  4x1  (列主序) -> 每个线程搬运 4行1列 的数据
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TA>{}, 
        Layout<Shape<_32, _8>, Stride<_1, _32>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape< _4, _1>>{}                   // Value  Layout: M-major (ColMajor)
    );
```

如下所示使用一个线程对32x32的地址空间进行搬运，拷贝的流程如下所示

- 生成copyEngine
- 获取当前线程对应的切片
- 按照切片逻辑去src和dst的tensor中进行切片
- 使用copy函数进行数据拷贝

```cpp
__global__ void one_thread_copy_kernel(const TA* g_in, TA* g_out, int M, int N) {
    // __shared__ TA smem[32*32];
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TA>{}, 
        Layout<Shape<_1, _1>, Stride<_1, _1>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape<_32,_32>>{}                   // Value  Layout: M-major (ColMajor)
    );
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        auto thr_copy = copyA.get_thread_slice(threadIdx.x);
        Tensor gA_src = make_tensor(make_gmem_ptr(g_in), make_shape(M, N));
        Tensor gA_dst = make_tensor(make_gmem_ptr(g_out), make_shape(M, N)); 
        Tensor tAgA_src = thr_copy.partition_S(gA_src);
        Tensor tAgA_dst = thr_copy.partition_D(gA_dst);
        copy(copyA, tAgA_src, tAgA_dst);
    }
}
```

如果是涉及到block层级，那么需要先在block层级进行划分，在block层级得到local_tile，再按照上面的思路进行线程负载的划分拷贝

```cpp
template <int bM, int bN>
__global__ void grid_block_copy_ce(float* in, float* out, int M, int N) {
    auto layout = make_layout(make_shape(M, N), GenRowMajor{});
    auto gIn = make_tensor(make_gmem_ptr(in), layout);
    auto gOut = make_tensor(make_gmem_ptr(out), layout);

    auto tile_shape = make_shape(Int<bM>{}, Int<bN>{});
    auto tile_coord = make_coord(blockIdx.x, blockIdx.y);

    auto gIn_b = local_tile(gIn, tile_shape, tile_coord);
    auto gOut_b = local_tile(gOut, tile_shape, tile_coord);

    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{}, 
        Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape<_1,_4>>{}                   // Value  Layout: M-major (ColMajor)
    );
    auto thr_copy = copyA.get_slice(threadIdx.x);
    auto tgIn = thr_copy.partition_S(gIn_b);   // Source (源) 切分
    auto tgOut = thr_copy.partition_D(gOut_b); // Destination (目标) 切分
    copy(tgIn, tgOut);
}
```