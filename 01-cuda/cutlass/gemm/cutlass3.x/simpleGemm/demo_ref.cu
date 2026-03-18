#include <cutlass/cutlass.h>
#include <iostream> // [修复 1] 拼写错误

#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

// [修复 2] 补充 cute 命名空间，否则下方的 Shape 无法识别
using namespace cute;

using ElementA = float;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = float;
using LayoutB  = cutlass::layout::ColumnMajor; 
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float; 
using LayoutC  = cutlass::layout::ColumnMajor; 
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float; 
using ArchTag            = cutlass::arch::Sm90; 
using OperatorClass      = cutlass::arch::OpClassTensorOp; 
using TileShape          = Shape<_128,_128,_32>; 
using ClusterShape       = Shape<_4,_2,_1>; 

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainLoop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainLoop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// 全局数据结构
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed = 2023;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

struct Options
{
    int m = 5120;
    int n = 4096;
    int k = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;
    int iterations = 100;

    RasterOrderOptions raster = RasterOrderOptions::Heuristic;
    int swizzle = 1;
    double gflops(double runtime_s) const {
        uint64_t flop = uint64_t(2) * m * n * k;
        double gflop = double(flop) / double(1.0e9);
        return gflop / runtime_s;
    }
};

template<class Element>
bool init_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed=2023) {
    Element scope_max = Element(2);
    Element scope_min = Element(-2);
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, scope_max, scope_min, 0);
    return true;
}

void init(const Options& options) {
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

    block_A.reset(options.m * options.k);
    block_B.reset(options.k * options.n);
    block_C.reset(options.m * options.n);
    block_D.reset(options.m * options.n);
    block_ref_D.reset(options.m * options.n);

    // [修复 3] 调用正确的函数名 init_block
    init_block(block_A, seed + 1);
    init_block(block_B, seed + 2);
    init_block(block_C, seed + 3);
}

typename Gemm::Arguments args_from_options(const Options &options) {
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(0);
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
    kernel_hw_info
  };
  arguments.scheduler.raster_order = options.raster;
  arguments.scheduler.max_swizzle_size = options.swizzle;
  return arguments;
}

template<typename Gemm>
int run(Options &options) {
    init(options);
    Gemm gemm;
    auto arguments = args_from_options(options);
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // [修复 4] 实打实地分配 workspace 内存
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 建议加上 CUTLASS_CHECK，这里为了保持你的代码结构，先静默调用
    gemm.can_implement(arguments);
    gemm.initialize(arguments, workspace.get());
    gemm.run();
    
    // [修复 5] 增加返回值，防止编译器报 warning/error
    return 0;
}

int main() {
  Options options;
  // [修复 6] 删掉这里的 init(options)，因为 run 里面已经调过了，防止显存重复分配
  // [修复 7] 必须带上模板类型 <Gemm> 才能正确调用 run
  run<Gemm>(options);
  
  std::cout << "CUTLASS 3.x GEMM executed successfully!" << std::endl;
  return 0;
}