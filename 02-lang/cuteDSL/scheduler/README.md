# CUTLASS GEMM 的 Tile Scheduler

Tile scheduler 决定 CTA/cluster 如何取得输出 tile（以及在部分算法中如何分配 K 维工作）。它不决定 CTA 内部怎样做 MMA、搬运数据或执行 epilogue；这些由 mainloop、tile/cluster shape 和 kernel schedule 等配置决定。

## 常见调度方式

| 调度方式 | 工作分配 | 适合场景 | 代价/注意点 |
| --- | --- | --- | --- |
| 普通、非持久化（individual tile） | grid 中的 CTA 直接对应输出矩阵上的 tile；一个 CTA 通常算一个输出 tile。 | tile 数量充足、负载均匀，或简单基准实现。 | tile 很多时会有大量 CTA；每个 CTA 都要走一次 kernel 的启动与收尾路径。 |
| Persistent | 只启动一批常驻 CTA/cluster；每个 CTA 完成一个 tile 后，再按确定规则取下一个 tile，直到没有 tile。 | 大矩阵、希望控制并发 CTA 数量，或使用 TMA/warp-specialized 等需要 persistent 工作循环的内核。 | 调度通常很轻，但固定的 tile 分配顺序在负载不均时可能不够灵活。CUTLASS 的 `PersistentScheduler` 是静态 persistent 调度。 |
| Stream-K | 把 GEMM 的 K 维迭代工作切成 work unit 分配给 CTA，使 CTA 数量不足以填满 GPU 时仍能并行；必要时多个 CTA 的部分结果要归约。 | 输出 tile 数少、K 很长，普通按输出 tile 并行不能充分占满 GPU；也可改善某些形状下的尾部负载不均。 | 可能增加部分和归约、同步和临时 workspace 开销；并非对所有形状都更快，而且受 kernel schedule/架构支持约束。 |
| CLC 动态 persistent（SM100/Blackwell） | 常驻 cluster 通过 Cluster Launch Control 动态取得下一块工作；已完成的 cluster 可接手尚未开始的 CTA 工作。 | 动态负载、CTA 执行时间差异较大，希望运行时重新分配工作。 | 需要 Blackwell CLC 支持和相应 kernel 实现；不是把旧 kernel 的 scheduler tag 换掉就能启用。 |

上表里的“普通/非持久化”描述的是 tile 到 grid 的直接映射；CUTLASS 3.x 中常由 scheduler 选择器根据 kernel 配置决定实际实现。`cutlass::gemm::PersistentScheduler` 与 `cutlass::gemm::StreamKScheduler` 是显式调度 tag。SM90 上常见的 TMA warp-specialized GEMM 通常采用 persistent tile scheduler。具体可用组合由 CUTLASS 版本、架构及 mainloop/epilogue schedule 决定。

## Raster order 不是另一种 scheduler

Persistent scheduler 还可以选择沿 M 或 N 的 tile 遍历顺序。它改变相邻 CTA 访问输出 tile 的顺序，可能影响 A/B 矩阵数据的缓存复用和访存局部性；它不会改变 GEMM 数值结果。`Heuristic` 让 CUTLASS 根据问题和配置选择顺序，`max_swizzle_size` 控制可用的 tile 排列 swizzle 范围。实际收益依赖矩阵形状、tile/cluster shape 和 GPU，通常需要测量。

当前目录相邻的 [`simpleGemm/demo.cu`](../../../01-cuda/cutlass/gemm/cutlass3.x/simpleGemm/demo.cu) 就配置了这些参数：

```cpp
using RasterOrderOptions = typename
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

arguments.scheduler.raster_order = RasterOrderOptions::Heuristic;
arguments.scheduler.max_swizzle_size = 4;
```

`detail::PersistentTileSchedulerSm90Params` 属于实现细节；项目若切换 CUTLASS 版本，应以该版本的 scheduler 参数定义为准。

## 用一个 GEMM 类型切换 scheduler

下面展示类型层的最小改动。`CollectiveMainloop`、`CollectiveEpilogue` 及 `Gemm::Arguments` 初始化沿用现有 Hopper 示例：

```cpp
using ProblemShape = cute::Shape<int, int, int>; // M, N, K

template <class Scheduler>
using GemmKernelFor = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    Scheduler>;

// A. Persistent：显式指定静态 persistent scheduler
using PersistentKernel = GemmKernelFor<cutlass::gemm::PersistentScheduler>;
using PersistentGemm = cutlass::gemm::device::GemmUniversalAdapter<PersistentKernel>;

// B. Stream-K：让多个 CTA 可以协作处理 K 维工作
using StreamKKernel = GemmKernelFor<cutlass::gemm::StreamKScheduler>;
using StreamKGemm = cutlass::gemm::device::GemmUniversalAdapter<StreamKKernel>;

// C. Individual / 非持久化：不传显式 scheduler tag，保留 CUTLASS 默认选择
using DefaultKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using DefaultGemm = cutlass::gemm::device::GemmUniversalAdapter<DefaultKernel>;
```

把 `PersistentGemm` 或 `StreamKGemm` 传给现有的 `run<Gemm>()` 即可尝试对应类型，例如：

```cpp
run<PersistentGemm>(options);
// 或 run<StreamKGemm>(options); // 先确认当前 kernel schedule 支持 Stream-K
```

这里的 `DefaultGemm` **不能保证**一定是 individual scheduler：省略 scheduler tag 表示让该版本的 `TileSchedulerSelector` 按 kernel schedule 选择实现。需要明确控制调度时，使用显式 tag，并确认对应 kernel 能实例化、`can_implement()` 成功。CUTLASS 官方 Hopper collective builder 示例也将 `PersistentScheduler` 作为 tile scheduler 类型参数，并展示了 Stream-K 配置。

## 选型直觉

- 输出 tile 很多且工作均匀：先用 CUTLASS 默认/普通映射作为基线。
- 大型、规则的矩阵乘：比较 persistent，并尝试 `AlongM`、`AlongN` 和 `Heuristic` raster 顺序。
- 输出 tile 数偏少而 K 很长：测试 Stream-K，确认收益足以覆盖归约成本。
- Blackwell 上有明显动态负载不均：再考虑 CLC 动态 persistent 的专用实现。

tile scheduler 不是固定的性能等级排序。同一策略在不同的 M/N/K、tile shape、cluster shape 和 GPU 上可能表现不同；应使用目标 workload 实测。

## CuTe Python DSL 中的 scheduler

CuTe DSL 里也有 scheduler 概念，但表达方式是 `WorkQueue` 和 schedule task，不是上面的 C++ scheduler tag。主要有三种调度形态：

1. **Non-persistent**：launch grid 直接映射到逻辑 tile，没有 `WorkQueue`。
2. **Static persistent**：CTA/任务通过本地算术推进共享的工作序列，没有单独的 scheduler task。
3. **Dynamic persistent（CLC）**：单独的 scheduler task 从硬件 CLC 获取 tile，data task 消费队列中的 tile。

静态 persistent 的结构示意（省略具体的 load/MMA/store）：

```python
@schedule
def data_task(wq: WorkQueue):
    with work_tile_loop(wq):
        with domain_loop(0, num_k_tiles, 1):
            compute_one_k_tile()

        # 每完成一个 work tile，所有参与任务都要按相同边界推进队列
        wq.try_wait()
        wq.wait()
        wq.get_and_advance_work_tile()
        wq.release()
```

动态 persistent 额外需要 scheduler task 来取工作：

```python
@schedule
def scheduler_task(wq: WorkQueue):
    with work_tile_loop(wq):
        wq.try_acquire()
        wq.acquire()
        wq.fetch_work_tile()
        wq.commit()
        wq.try_wait()
        wq.wait()
        wq.get_and_advance_work_tile()
        wq.release()
```

实际 DSL kernel 还必须正确声明任务间依赖，并保证每个 persistent task 在每个 work tile 的尾部推进队列一次。静态模式下不需要 `scheduler_task`；动态 CLC 模式下 data task 的队列消费尾部仍然存在。不要把这套 DSL task/WorkQueue API 和 C++ `PersistentScheduler` / `StreamKScheduler` tag 当成可直接互换的写法。

## 参考

- [CUTLASS Hopper GEMM with Collective Builder（官方示例 49）](https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu)：示例模板含 `PersistentScheduler`，并实例化 Stream-K 配置。
- [CUTLASS SM90 tile scheduler 实现](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp)：persistent scheduler、raster order 等实现。
- [CUTLASS SM90 Stream-K scheduler 实现](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp)：Stream-K work tile 和归约相关逻辑。
- [CUTLASS Blackwell Cluster Launch Control 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)：SM100 动态 persistent 调度概念。
- [CuTe DSL Schedules 文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_schedules.html)：non-persistent、static persistent 和 CLC dynamic persistent 的 WorkQueue 结构。
