
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

using namespace cute;

// ============================================================================
// Shared Memory Storage
// ============================================================================
template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
    // 128字节对齐，满足TMA要求
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_B;

    // mbarrier 的大小刚好为64位，size<2>取出pipe的层级
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

// ============================================================================
// WGMMA GEMM Kernel (TMA + Pipeline)
// ============================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
    // 静态断言检查
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //
    auto [M, N, K] = shape_MNK;

    // 通过TMA descriptor获取全局tensor
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                   // (M,K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));                   // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);      // (M,N)

    // 获取当前CTA对应的tile
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    //
    // TMA Partition
    //
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sB), group_modes<0, 2>(gB));

    // TMA传输的字节数
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                        + sizeof(make_tensor_like(tensor<0>(tBsB)));

    //
    // PREFETCH - 流水线预取
    //
    auto K_PIPE_MAX = size<1>(tAsA);

    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    // 由一个线程获取到warpID然后广播到其他线程
    int warp_idx = cutlass::canonical_warp_idx_sync();
    // 选择出一个线程
    int lane_predicate = cute::elect_one_sync();

    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA bar
    using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA bar

    // 初始化barriers
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }

    // 确保所有CTA的barrier初始化完成
    cluster_sync();

    // 启动异步加载，填满所有pipe
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    //
    // MMA Partition
    //
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // 累加器
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    // MMA Descriptor fragments (直接从SMEM读取，无需拷贝到寄存器)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

    //
    // PIPELINED MAIN LOOP
    //
    // TODO: 当前实现存在明显的流水线气泡，后续优化方向：
    //   1. Warp Specialization: 将 Producer(TMA) 和 Consumer(MMA) 分配到不同的 Warp Group，
    //      使 TMA 搬运和 WGMMA 计算真正并行（当前同一 WG 串行执行导致无法 overlap）
    //   2. warpgroup_wait<N>: 当前 wait<0> 要求所有 MMA 完成才继续，改为 wait<1> 或 wait<2>
    //      允许多个 MMA batch 重叠执行
    //   3. 增加 Pipeline 深度: 当前 3 级 → 4~7 级，更好隐藏 TMA 延迟
    //   4. Persistent Kernel: CTA 复用，减少 launch 开销和 tail effect
    //   5. Epilogue 融合: 将 axpby 融合到最后一轮 MMA 中减少全局内存访问
    //
    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
        // 等待Producer（TMA）完成数据加载
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        // WGMMA计算
        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
        warpgroup_commit_batch();

        // 等待所有MMA完成
        warpgroup_wait<0>();  // TODO: 改为 wait<1> 以允许 MMA overlap，需配合调整 barrier 逻辑

        // 通知Consumer已消费完毕
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        // 发起新的TMA拷贝（如果还有tile需要加载）
        if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0)) {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    //
    // Epilogue: C = alpha * accum + beta * C
    //
    axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================
// Host-side GEMM launch (NT layout: A=col-major, B=row-major)
// ============================================================================
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK) col-major
    auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK) col-major (transposed)
    auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN) col-major

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<  3>{};  // Pipeline depth

    // Define smem layouts with swizzle (3D: M/N, K, PIPE)
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // Define the WGMMA (64x64x16 FP16)
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    // Create Global memory tensors for TMA inspection
    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    // Create TMA descriptors
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    //
    // Setup and Launch
    //
    int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
                 round_up(size(ceil_div(n, bN)), dimCluster.y));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
                                &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                             TA, decltype(sA), decltype(tmaA),
                                             TB, decltype(sB), decltype(tmaB),
                                             TC, decltype(dC), decltype(tiled_mma),
                                             decltype(alpha), decltype(beta)>);

    // 设置动态shared memory大小
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // 启动kernel
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                               prob_shape, cta_tiler,
                                                               A, tmaA,
                                                               B, tmaB,
                                                               C, dC, tiled_mma,
                                                               alpha, beta);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "Error: Failed at kernel launch\n");
    }
}

// ============================================================================
// cuBLAS HGEMM wrapper
// ============================================================================
void cublas_hgemm(cublasHandle_t handle,
                  int m, int n, int k,
                  half const* A, int ldA,
                  half const* B, int ldB,
                  half      * C, int ldC)
{
    // C = A * B^T  (NT layout)
    // cuBLAS col-major: C = alpha * op(A) * op(B) + beta * C
    // op(A) = A (N), op(B) = B^T (T)
    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    cublasStatus_t stat = cublasHgemm(handle,
                                       CUBLAS_OP_N, CUBLAS_OP_T,
                                       m, n, k,
                                       &alpha_h,
                                       A, ldA,
                                       B, ldB,
                                       &beta_h,
                                       C, ldC);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS HGEMM failed: %d\n", stat);
    }
}

// ============================================================================
// Main: Benchmark WGMMA vs cuBLAS
// ============================================================================
int main(int argc, char** argv)
{
    // 检查GPU架构
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);

    if (props.major != 9) {
        printf("This example requires NVIDIA Hopper GPU (SM90)\n");
        printf("Detected: %s (SM%d%d)\n", props.name, props.major, props.minor);
        return 0;
    }

    printf("GPU: %s\n\n", props.name);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    // 解析命令行参数
    int m = 4096;
    if (argc >= 2) sscanf(argv[1], "%d", &m);

    int n = 4096;
    if (argc >= 3) sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4) sscanf(argv[3], "%d", &k);

    printf("Problem Size: M=%d, N=%d, K=%d\n", m, n, k);
    printf("Data Type: FP16\n");
    printf("Layout: A=col-major(N), B=col-major(T)\n\n");

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;

    TI alpha = TI(1.0f);
    TI beta  = TI(0.0f);

    // 分配并初始化矩阵
    int ldA = m;
    int ldB = n;
    int ldC = m;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n, TC(0));

    srand(42);
    for (int j = 0; j < m * k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n * k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C_wgmma = h_C;
    thrust::device_vector<TC> d_C_cublas(m * n, TC(0));

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int warmup_iterations = 10;
    const int timing_iterations = 100;
    GPU_Clock timer;

    // ========================================================================
    // 1. WGMMA GEMM
    // ========================================================================
    printf("--- WGMMA GEMM (CuTe + TMA) ---\n");

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        gemm_nt(m, n, k, alpha,
                d_A.data().get(), ldA,
                d_B.data().get(), ldB,
                beta,
                d_C_wgmma.data().get(), ldC);
    }
    CUTE_CHECK_LAST();

    // 正式计时
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm_nt(m, n, k, alpha,
                d_A.data().get(), ldA,
                d_B.data().get(), ldB,
                beta,
                d_C_wgmma.data().get(), ldC);
    }
    double wgmma_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();

    double wgmma_tflops = gflops / wgmma_time / 1000.0;
    printf("  Time:      %.4f ms\n", wgmma_time * 1000);
    printf("  Perf:      %.1f GFlop/s  (%.2f TFlop/s)\n", gflops / wgmma_time, wgmma_tflops);

    // ========================================================================
    // 2. cuBLAS HGEMM
    // ========================================================================
    printf("\n--- cuBLAS HGEMM ---\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        cublas_hgemm(handle, m, n, k,
                     reinterpret_cast<half const*>(d_A.data().get()), ldA,
                     reinterpret_cast<half const*>(d_B.data().get()), ldB,
                     reinterpret_cast<half*>(d_C_cublas.data().get()), ldC);
    }
    cudaDeviceSynchronize();

    // 正式计时
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        cublas_hgemm(handle, m, n, k,
                     reinterpret_cast<half const*>(d_A.data().get()), ldA,
                     reinterpret_cast<half const*>(d_B.data().get()), ldB,
                     reinterpret_cast<half*>(d_C_cublas.data().get()), ldC);
    }
    double cublas_time = timer.seconds() / timing_iterations;

    double cublas_tflops = gflops / cublas_time / 1000.0;
    printf("  Time:      %.4f ms\n", cublas_time * 1000);
    printf("  Perf:      %.1f GFlop/s  (%.2f TFlop/s)\n", gflops / cublas_time, cublas_tflops);

    // ========================================================================
    // 3. 正确性验证
    // ========================================================================
    printf("\n--- Correctness Check ---\n");

    thrust::host_vector<TC> h_wgmma  = d_C_wgmma;
    thrust::host_vector<TC> h_cublas = d_C_cublas;

    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float diff = fabs(float(h_wgmma[i]) - float(h_cublas[i]));
        max_error = fmax(max_error, diff);
        avg_error += diff;
    }
    avg_error /= (m * n);

    printf("  Max Error: %.6f\n", max_error);
    printf("  Avg Error: %.6f\n", avg_error);
    if (max_error < 1.0f) {
        printf("  Status:    PASSED\n");
    } else {
        printf("  Status:    FAILED (max error too large)\n");
    }

    // ========================================================================
    // 4. 性能对比
    // ========================================================================
    printf("\n============================================\n");
    printf("Performance Comparison:\n");
    printf("  WGMMA:   %8.1f GFlop/s  (%6.2f TFlop/s)  %.4f ms\n",
           gflops / wgmma_time, wgmma_tflops, wgmma_time * 1000);
    printf("  cuBLAS:  %8.1f GFlop/s  (%6.2f TFlop/s)  %.4f ms\n",
           gflops / cublas_time, cublas_tflops, cublas_time * 1000);
    printf("  Ratio (WGMMA/cuBLAS): %.1f%%\n",
           100.0 * (gflops / wgmma_time) / (gflops / cublas_time));
    printf("============================================\n");

    cublasDestroy(handle);

#else
    printf("CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it is not.\n");
    printf("Please compile with -DCUTLASS_ARCH_MMA_SM90_SUPPORTED or arch=sm_90a.\n");
#endif

    return 0;
}
