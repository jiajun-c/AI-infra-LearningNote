# cutlass 3.x

cutlass3.x 中将 cutlass2.x中关于layout，stride，迭代器等一套复杂的体系全部都适用 cute 的 layout进行处理

下面这张图就很形象地展示了这一点

![alt text](image.png)

cutlass 3.x的函数签名

```cpp
// 3.x 的 Arguments 构造：高度模块化、多维代数化
GemmUniversal::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, // 告诉它这是普通 GEMM 还是 Batched
    {M, N, K},                               // 问题规模 (ProblemShape)
    { // Mainloop 参数区 (管进)
        d_A, stride_A,                       // 注意：不再是 lda，而是 CuTe 的多维 Stride
        d_B, stride_B
    },
    { // Epilogue 参数区 (管出)
        {alpha, beta},                       // 算完后的操作参数
        d_C, stride_C,
        d_D, stride_D
    }
};
```

gemm的层级分为下面的几种
- Device：`cutlass::gemm::device::GemmUniversalAdapter`
- Kernel: `cutlass::gemm::kernel::GemmUniversal`
- Collective: 
    - `cutlass::gemm::collective::CollectiveMma`
    - `cutlass::epilogue::collective::DefaultEpilogue`
    - `cutlass::epilogue::collective::Epilogue`
- Tiled (MMA and Copy)
    - `cute::TiledMma` and `cute::TiledCopy`
    - `cute::gemm()` and `cute::copy()`

如果要实例化一个矩阵乘法，其顺序为
- 编写包括主循环和后处理部分
- 把他们组合起来去构建一个kernel类型
- 包装kernel为一个device层级的适配器

如下所示第一步是建立主循环，第二步是建立epilogue，第三步组合成一个GemmKernel，最后暴露成一个device侧的GemmHandle

```cpp
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```