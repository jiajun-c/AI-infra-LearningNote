# Collective API

Collective是一个mma矩阵乘法和拷贝指令是如何分块的集合，通过这个集合我们可以了解到

## Collective Mainloops

Mainloops一般指的是我们矩阵运算上的主循环，例如在MNK的矩阵计算上对K维度的一个遍历


如下所示是对CollectiveMma

```cpp
namespace cutlass::gemm::collective {

template <
  class DispatchPolicy,// 调度策略
  class TileShape, // 分块形状
  class ElementA,  // A 和 B的数据类型和stride
  class StrideA,
  class ElementB,
  class StrideB,
  class TiledMma,
  class GmemTiledCopyA, // 全局到共享的搬运工，以及共享内存的布局
  class SmemLayoutAtomA,
  class SmemCopyAtomA,
  class TransformA,
  class GmemTiledCopyB,
  class SmemLayoutAtomB,
  class SmemCopyAtomB,
  class TransformB
>
struct CollectiveMma {
  static_assert(sizeof(ElementA) == 0, "Could not find a mainloop specialization.");
};

} // namespace cutlass::gemm::collective
```

## Collectivate Dispatch Policies

指定流水的层级，cluster的shape，Warp 特化调度策略

```cpp
// n-buffer in smem (Hopper TMA),
// pipelined with Hopper GMMA and TMA,
// warp-specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};
```

调度算法大致分为下面的几种，CpAsync和tma表示不同的数据拷贝方案，WarpSpecialized表示将线程分为生产者和消费者，如果不指定，那么就是所有的线程做一样的事情，难以去隐藏延迟。对于后缀其表示计算阵型调度
- pingpong调度：分配 2 个 Consumer Warp Groups。当 Group 1 在猛烈计算第 N 块数据时，Group 2 在后台静静等待 Producer 把第 N+1 块数据搬进来；等 Group 1 算完，Group 2 立刻无缝衔接开火
- cooperative（协同调度）：不再让两组 Warp 交替执行，而是让所有的 Consumer Warps 像群狼一样，共同扑向同一个超大的矩阵块

```cpp
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperative { };
```

可以手动选择`CollectiveMma`引擎，也可以使用`CollectiveBuilder`来根据输入的参数自动选择最优的方案

