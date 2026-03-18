#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cublas_v2.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
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

// --------------------------------------------------------------------------
// 内联工具宏与 GpuTimer（替代 helper.h，避免模板内名称查找问题）
// --------------------------------------------------------------------------
#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: "                                       \
                << cutlassGetStatusString(error)                               \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)        \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUBLAS_CHECK(status)                                                   \
  {                                                                            \
    cublasStatus_t error = status;                                             \
    if (error != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "Got cublas error: " << (int)error                          \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

struct GpuTimer {
  cudaEvent_t _start, _stop;
  GpuTimer()  { cudaEventCreate(&_start); cudaEventCreate(&_stop); }
  ~GpuTimer() { cudaEventDestroy(_start); cudaEventDestroy(_stop); }
  void start(cudaStream_t s = 0) { cudaEventRecord(_start, s); }
  void stop()                    { cudaEventRecord(_stop, 0);   }
  float elapsed_millis() {
    float elapsed = 0.0f;
    cudaEventSynchronize(_stop);
    cudaEventElapsedTime(&elapsed, _start, _stop);
    return elapsed;
  }
};

using namespace cute;
#define CUTLASS_ARCH_MMA_SM90_SUPPORTED
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// --------------------------------------------------------------------------
// 1. GEMM 内核配置 (CUTLASS 3.x 核心区)
// --------------------------------------------------------------------------
using ElementA = float; 
using LayoutA  = cutlass::layout::RowMajor; 
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

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
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
    Shape<int,int,int>,
    CollectiveMainloop,
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

// --------------------------------------------------------------------------
// 2. 参数配置结构（支持命令行解析）
// --------------------------------------------------------------------------
// 用法: ./demo [-m M] [-n N] [-k K] [-alpha A] [-beta B] [-iterations I]
struct Options {
  int m = 5120;
  int n = 4096;
  int k = 4096;
  float alpha = 1.0f;
  float beta  = 0.0f;
  int iterations = 100; // 测速循环次数

  // 调度器参数
  RasterOrderOptions raster = RasterOrderOptions::Heuristic;
  int swizzle = 1;

  double gflops(double runtime_s) const {
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }

  void parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg(argv[i]);
      if ((arg == "-m" || arg == "--m") && i + 1 < argc) {
        m = std::atoi(argv[++i]);
      } else if ((arg == "-n" || arg == "--n") && i + 1 < argc) {
        n = std::atoi(argv[++i]);
      } else if ((arg == "-k" || arg == "--k") && i + 1 < argc) {
        k = std::atoi(argv[++i]);
      } else if ((arg == "-alpha" || arg == "--alpha") && i + 1 < argc) {
        alpha = std::atof(argv[++i]);
      } else if ((arg == "-beta" || arg == "--beta") && i + 1 < argc) {
        beta = std::atof(argv[++i]);
      } else if ((arg == "-iterations" || arg == "--iterations") && i + 1 < argc) {
        iterations = std::atoi(argv[++i]);
      } else if (arg == "-h" || arg == "--help") {
        std::cout << "CUTLASS 3.x SM90 GEMM Benchmark\n"
                  << "用法: ./demo [选项]\n"
                  << "  -m <int>          矩阵 M 维度 (默认: 5120)\n"
                  << "  -n <int>          矩阵 N 维度 (默认: 4096)\n"
                  << "  -k <int>          矩阵 K 维度 (默认: 4096)\n"
                  << "  -alpha <float>    alpha 系数   (默认: 1.0)\n"
                  << "  -beta <float>     beta 系数    (默认: 0.0)\n"
                  << "  -iterations <int> 测速迭代次数 (默认: 100)\n"
                  << "  -h, --help        显示帮助信息\n";
        exit(0);
      }
    }
  }
};

// --------------------------------------------------------------------------
// 3. GEMM 初始化与执行逻辑
// --------------------------------------------------------------------------
template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed=2023) {
  Element scope_max = Element(2);
  Element scope_min = Element(-2);
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);
  return true;
}

void initialize(const Options &options) {
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 1);
  initialize_block(block_B, seed + 2);
  initialize_block(block_C, seed + 3);
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

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  DeviceGemmReference gemm_reference;
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha), ref_A, ref_B,
    ElementAccumulator(options.beta), ref_C, ref_D);

  CUDA_CHECK(cudaDeviceSynchronize());
  return cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());
}

template <typename Gemm>
int run(Options &options) {
  initialize(options);
  Gemm gemm;
  auto arguments = args_from_options(options);
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  (gemm.can_implement(arguments));
  (gemm.initialize(arguments, workspace.get()));
  (gemm.run());

  bool passed = verify(options);
  std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
  if (!passed) exit(-1);

  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      (gemm.initialize(arguments, workspace.get())); // TMA 需要每次重置状态
      (gemm.run());
    }
    timer.stop();

    double avg_runtime_ms = double(timer.elapsed_millis()) / double(options.iterations);
    double gflops = options.gflops(avg_runtime_ms / 1000.0);

    std::cout << "Problem Size: " << options.m << "x" << options.n << "x" << options.k << std::endl;
    std::cout << "Avg runtime: " << avg_runtime_ms << " ms\n";
    std::cout << "GFLOPS: " << gflops << std::endl;
  }
  return 0;
}

#endif

// --------------------------------------------------------------------------
// 5. cuBLAS 基准测试 (用于性能对比)
// --------------------------------------------------------------------------
// cuBLAS 执行 D = alpha * A * B + beta * C
// 注意：cuBLAS 默认列主序，需要根据 A/B 的布局设置 transa/transb
//
// 参数 cublas_output: 可选输出缓冲区指针，若非空则将 cuBLAS 计算结果写入该缓冲区
//                     用于与 CUTLASS 结果做正确性交叉验证
void run_cublas(Options &options, double &avg_runtime_ms_out, double &gflops_out,
                float *cublas_output = nullptr) {
  int m = options.m, n = options.n, k = options.k;
  float alpha = options.alpha, beta = options.beta;

  // 分配设备内存 — 直接复用全局 block_A/B/C 的数据，保证输入一致
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * m * k));
  CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * k * n));
  CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * m * n));
  CUDA_CHECK(cudaMalloc(&d_D, sizeof(float) * m * n));

  // 使用与 CUTLASS 相同的输入数据（全局 block_A/B/C），确保正确性验证有意义
  CUDA_CHECK(cudaMemcpy(d_A, block_A.get(), sizeof(float) * m * k, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, block_B.get(), sizeof(float) * k * n, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, block_C.get(), sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // cuBLAS 使用列主序。我们的矩阵布局：
  //   A: RowMajor    (m x k) => 在列主序视角下相当于 A^T (k x m), lda = k
  //   B: ColumnMajor (k x n) => 已经是列主序, ldb = k
  //   C: ColumnMajor (m x n) => 已经是列主序, ldc = m
  //   D: ColumnMajor (m x n) => 已经是列主序, ldd = m
  //
  // cuBLAS 计算 D = alpha * op(A_cublas) * op(B_cublas) + beta * C_cublas
  // 列主序下: D(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n)
  // 因为我们的 A 是 RowMajor，相当于列主序存储的转置矩阵
  // 所以: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, ...)
  //   A_cublas 存储: 列主序 k x m (即 RowMajor m x k), lda = k, op = T => 得到 m x k
  //   B_cublas 存储: 列主序 k x n, ldb = k, op = N => 得到 k x n

  // Warmup
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  CUBLAS_CHECK(cublasSgemm(handle,
    CUBLAS_OP_T,   // A 是 RowMajor => 列主序下转置
    CUBLAS_OP_N,   // B 是 ColumnMajor => 不转置
    m, n, k,
    &alpha,
    d_A, k,        // lda = k (RowMajor 的行宽)
    d_B, k,        // ldb = k (ColumnMajor 的行宽)
    &beta,
    d_C, m));       // ldc = m
  CUDA_CHECK(cudaDeviceSynchronize());

  // 先执行一次正式计算，将结果保存到 d_D 供正确性验证使用
  CUDA_CHECK(cudaMemcpy(d_D, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
  CUBLAS_CHECK(cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A, k,
    d_B, k,
    &beta,
    d_D, m));
  CUDA_CHECK(cudaDeviceSynchronize());

  // 若调用方需要 cuBLAS 的输出结果，拷贝到 cublas_output 缓冲区
  if (cublas_output) {
    CUDA_CHECK(cudaMemcpy(cublas_output, d_D, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
  }

  // 计时
  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      // 每次把 C 拷贝到 D, 再在 D 上做 gemm (与 cutlass 行为对齐)
      CUDA_CHECK(cudaMemcpy(d_D, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
      CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, k,
        d_B, k,
        &beta,
        d_D, m));
    }
    timer.stop();

    avg_runtime_ms_out = double(timer.elapsed_millis()) / double(options.iterations);
    gflops_out = options.gflops(avg_runtime_ms_out / 1000.0);
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_D));
}

// --------------------------------------------------------------------------
// 5.1 CUTLASS vs cuBLAS 正确性交叉验证
// --------------------------------------------------------------------------
// 比较 CUTLASS 输出 (block_D) 与 cuBLAS 输出，计算最大绝对误差和相对误差
// 使用 relative tolerance 方式，适合 float32 精度下大矩阵的比较
bool verify_cutlass_vs_cublas(const Options &options, float *d_cublas_output) {
  int num_elements = options.m * options.n;

  // 将 CUTLASS 和 cuBLAS 的输出拷回 Host
  std::vector<float> h_cutlass(num_elements);
  std::vector<float> h_cublas(num_elements);
  CUDA_CHECK(cudaMemcpy(h_cutlass.data(), block_D.get(),
                         sizeof(float) * num_elements, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_cublas_output,
                         sizeof(float) * num_elements, cudaMemcpyDeviceToHost));

  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  int    max_abs_idx  = 0;
  int    max_rel_idx  = 0;
  int    nan_count    = 0;
  int    inf_count    = 0;

  for (int i = 0; i < num_elements; ++i) {
    float a = h_cutlass[i];
    float b = h_cublas[i];

    if (std::isnan(a) || std::isnan(b)) { ++nan_count; continue; }
    if (std::isinf(a) || std::isinf(b)) { ++inf_count; continue; }

    double abs_diff = std::abs(double(a) - double(b));
    double denom    = std::max({std::abs(double(a)), std::abs(double(b)), 1e-6});
    double rel_diff = abs_diff / denom;

    if (abs_diff > max_abs_diff) { max_abs_diff = abs_diff; max_abs_idx = i; }
    if (rel_diff > max_rel_diff) { max_rel_diff = rel_diff; max_rel_idx = i; }
  }

  // float32 GEMM 在 k=4096 量级下，合理的相对误差阈值约 1e-3 ~ 1e-2
  // 使用 1e-2 作为宽松阈值（不同 kernel 的累加顺序不同，FP32 舍入会导致差异）
  constexpr double kRelTolerance = 1e-2;
  bool passed = (max_rel_diff < kRelTolerance) && (nan_count == 0);

  std::cout << "\n========================================" << std::endl;
  std::cout << " CUTLASS vs cuBLAS 正确性验证" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "  元素总数:     " << num_elements << std::endl;
  std::cout << "  NaN 个数:     " << nan_count << std::endl;
  std::cout << "  Inf 个数:     " << inf_count << std::endl;
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  最大绝对误差: " << max_abs_diff
            << " (索引 " << max_abs_idx
            << ", CUTLASS=" << h_cutlass[max_abs_idx]
            << ", cuBLAS=" << h_cublas[max_abs_idx] << ")" << std::endl;
  std::cout << "  最大相对误差: " << max_rel_diff
            << " (索引 " << max_rel_idx
            << ", CUTLASS=" << h_cutlass[max_rel_idx]
            << ", cuBLAS=" << h_cublas[max_rel_idx] << ")" << std::endl;
  std::cout << "  相对误差阈值: " << kRelTolerance << std::endl;
  std::cout << "  验证结果:     " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
  std::cout << std::fixed;

  return passed;
}

// --------------------------------------------------------------------------
// 6. Main 纯净入口 — CUTLASS vs cuBLAS 性能对比
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 9 || props.minor != 0) {
    std::cerr << "This example requires NVIDIA Hopper Architecture (compute capability 90).\n";
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  std::cout << "配置: M=" << options.m << " N=" << options.n << " K=" << options.k
            << " alpha=" << options.alpha << " beta=" << options.beta
            << " iterations=" << options.iterations << std::endl;

  // ---- CUTLASS SM90 GEMM ----
  double cutlass_avg_ms = 0.0, cutlass_gflops = 0.0;
  #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  {
    std::cout << "========================================" << std::endl;
    std::cout << " Running CUTLASS SM90 GEMM..." << std::endl;
    std::cout << "========================================" << std::endl;

    initialize(options);
    Gemm gemm;
    auto arguments = args_from_options(options);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    (gemm.can_implement(arguments));
    (gemm.initialize(arguments, workspace.get()));
    (gemm.run());

    bool passed = verify(options);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) exit(-1);

    if (options.iterations > 0) {
      GpuTimer timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        (gemm.initialize(arguments, workspace.get()));
        (gemm.run());
      }
      timer.stop();
      cutlass_avg_ms = double(timer.elapsed_millis()) / double(options.iterations);
      cutlass_gflops = options.gflops(cutlass_avg_ms / 1000.0);

      std::cout << "Problem Size: " << options.m << "x" << options.n << "x" << options.k << std::endl;
      std::cout << "Avg runtime:  " << cutlass_avg_ms << " ms" << std::endl;
      std::cout << "GFLOPS:       " << cutlass_gflops << std::endl;
    }
  }
  #endif

  // ---- cuBLAS SGEMM ----
  double cublas_avg_ms = 0.0, cublas_gflops = 0.0;
  // 分配缓冲区保存 cuBLAS 输出，用于与 CUTLASS 交叉验证
  cutlass::DeviceAllocation<float> cublas_output_buf(options.m * options.n);
  {
    std::cout << "\n========================================" << std::endl;
    std::cout << " Running cuBLAS SGEMM..." << std::endl;
    std::cout << "========================================" << std::endl;

    run_cublas(options, cublas_avg_ms, cublas_gflops, cublas_output_buf.get());

    std::cout << "Problem Size: " << options.m << "x" << options.n << "x" << options.k << std::endl;
    std::cout << "Avg runtime:  " << cublas_avg_ms << " ms" << std::endl;
    std::cout << "GFLOPS:       " << cublas_gflops << std::endl;
  }

  // ---- CUTLASS vs cuBLAS 正确性交叉验证 ----
  #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  {
    bool cross_check = verify_cutlass_vs_cublas(options, cublas_output_buf.get());
    if (!cross_check) {
      std::cerr << "\n  [WARNING] CUTLASS 与 cuBLAS 结果不一致，请检查布局/参数配置！" << std::endl;
    }
  }
  #endif

  // ---- 性能对比汇总表 ----
  std::cout << "\n========================================" << std::endl;
  std::cout << " 性能对比 (Problem: " << options.m << "x" << options.n << "x" << options.k
            << ", Iterations: " << options.iterations << ")" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  " << std::left << std::setw(18) << "Library"
            << std::right << std::setw(12) << "Avg (ms)"
            << std::setw(14) << "GFLOPS"
            << std::setw(14) << "Speedup" << std::endl;
  std::cout << "  " << std::string(58, '-') << std::endl;
  std::cout << "  " << std::left << std::setw(18) << "cuBLAS"
            << std::right << std::setw(12) << cublas_avg_ms
            << std::setw(14) << cublas_gflops
            << std::setw(14) << "1.000x (base)" << std::endl;
  double speedup = (cutlass_avg_ms > 0.0) ? (cublas_avg_ms / cutlass_avg_ms) : 0.0;
  std::cout << "  " << std::left << std::setw(18) << "CUTLASS SM90"
            << std::right << std::setw(12) << cutlass_avg_ms
            << std::setw(14) << cutlass_gflops
            << std::setw(14) << (std::to_string(speedup).substr(0,5) + "x") << std::endl;
  std::cout << std::endl;

  if (speedup > 1.0) {
    std::cout << "  => CUTLASS 比 cuBLAS 快 " << std::setprecision(1) << ((speedup - 1.0) * 100.0) << "%" << std::endl;
  } else if (speedup < 1.0 && speedup > 0.0) {
    std::cout << "  => cuBLAS 比 CUTLASS 快 " << std::setprecision(1) << ((1.0 / speedup - 1.0) * 100.0) << "%" << std::endl;
  } else {
    std::cout << "  => 两者性能基本持平" << std::endl;
  }

  return 0;
}