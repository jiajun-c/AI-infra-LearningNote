#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

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
    // 128字节对齐，满足 TMA 写入 Shared Memory 的严格要求
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_B;

    // mbarrier 大小刚好为64位，size<2>取出 pipe 的层级
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

// ============================================================================
// WGMMA GEMM Kernel (TMA + Pipeline + Predicated Epilogue)
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

    // Full and Tiled Tensors
    auto [M, N, K] = shape_MNK;

    // 通过 TMA descriptor 获取全局 tensor (TMA 硬件保证越界 K 维度自动补 0)
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                   // (M,K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));                   // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);      // (M,N)

    // 获取当前 CTA 对应的 tile
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

    // TMA Partition
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sB), group_modes<0, 2>(gB));

    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

    // PREFETCH - 流水线预取
    auto K_PIPE_MAX = size<1>(tAsA);
    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    // 初始化 barriers
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }
    cluster_sync();

    // 启动异步加载，填满所有 pipe
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

    // MMA Partition
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    // 累加器 (寄存器片段)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // MMA Descriptor fragments
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

    // PIPELINED MAIN LOOP
    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

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
// Host-side GEMM launch
// ============================================================================
template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_row_major(int m, int n, int k,
               Alpha alpha,
               TA const* A, int ldA,
               TB const* B, int ldB,
               Beta beta,
               TC      * C, int ldC,
               cudaStream_t stream = 0)
{
    auto M = int(m); auto N = int(n); auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // ==========================================
    // 护法 1: Row-Major Strides (行主序，K为连续维)
    // ==========================================
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});

    auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int< 64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<  3>{};

    // ==========================================
    // 护法 2: 共享内存 Swizzle 调整为 Layout_K
    // ==========================================
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // ==========================================
    // 护法 3: WGMMA 指令类型匹配 Major::K
    // ==========================================
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

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

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr, prob_shape, cta_tiler,
                                                               A, tmaA, B, tmaB, C, dC, tiled_mma, alpha, beta);
    CUTE_CHECK_LAST();
    if (status != cutlass::Status::kSuccess) { fprintf(stderr, "Error: Failed at kernel launch\n"); }
}

// ============================================================================
// cuBLAS HGEMM wrapper (Row-Major)
// ============================================================================
void cublas_hgemm_row_major(cublasHandle_t handle, int m, int n, int k,
                            half const* A, int ldA, half const* B, int ldB, half* C, int ldC)
{
    // C_row = A_row * B_row^T
    // 在 cuBLAS (Col-Major 视角) 下，等价于计算 C_col = B_col * A_col^T
    // 因此传入 cuBLAS 的矩阵顺序为 B 然后 A，并且对 B 使用 CUBLAS_OP_T
    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    cublasStatus_t stat = cublasHgemm(handle,
                                      CUBLAS_OP_T, CUBLAS_OP_N, 
                                      n, m, k, 
                                      &alpha_h,
                                      B, ldB, 
                                      A, ldA, 
                                      &beta_h,
                                      C, ldC);
    if (stat != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS failed\n"); }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);

    if (props.major != 9) {
        printf("Requires NVIDIA Hopper GPU (SM90)\n");
        return 0;
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    int m = 1023; if (argc >= 2) sscanf(argv[1], "%d", &m);
    int n = 1024; if (argc >= 3) sscanf(argv[2], "%d", &n);
    int k = 1024; if (argc >= 4) sscanf(argv[3], "%d", &k);

    printf("GPU: %s\n", props.name);
    printf("Problem Size: M=%d, N=%d, K=%d\n", m, n, k);
    printf("Layout: A=row-major, B=row-major (C = A * B^T)\n\n");

    using TA = cute::half_t; using TB = cute::half_t; using TC = cute::half_t;

    // ==========================================
    // 护法 4: Host 端强制 16 字节对齐分配
    // ==========================================
    int ldA = (k + 7) / 8 * 8; 
    int ldB = (k + 7) / 8 * 8; 
    int ldC = (n + 7) / 8 * 8; 

    thrust::host_vector<TA> h_A(m * ldA);
    thrust::host_vector<TB> h_B(n * ldB);
    thrust::host_vector<TC> h_C(m * ldC, TC(0));

    srand(42);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) { h_A[i * ldA + j] = TA(int((rand() % 2) ? 1 : -1)); }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) { h_B[i * ldB + j] = TB(int((rand() % 2) ? 1 : -1)); }
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C_wgmma = h_C;
    thrust::device_vector<TC> d_C_cublas(m * ldC, TC(0));

    double gflops = (2.0 * m * n * k) * 1e-9;
    GPU_Clock timer;

    // --- WGMMA ---
    for (int i = 0; i < 10; ++i) {
        gemm_row_major(m, n, k, TA(1.f), d_A.data().get(), ldA, d_B.data().get(), ldB, TA(0.f), d_C_wgmma.data().get(), ldC);
    }
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < 100; ++i) {
        gemm_row_major(m, n, k, TA(1.f), d_A.data().get(), ldA, d_B.data().get(), ldB, TA(0.f), d_C_wgmma.data().get(), ldC);
    }
    double wgmma_time = timer.seconds() / 100.0;
    printf("  WGMMA Time: %.4f ms, %.1f GFlop/s\n", wgmma_time * 1000, gflops / wgmma_time);

    // --- cuBLAS ---
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < 10; ++i) {
        cublas_hgemm_row_major(handle, m, n, k, (half*)d_A.data().get(), ldA, (half*)d_B.data().get(), ldB, (half*)d_C_cublas.data().get(), ldC);
    }
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < 100; ++i) {
        cublas_hgemm_row_major(handle, m, n, k, (half*)d_A.data().get(), ldA, (half*)d_B.data().get(), ldB, (half*)d_C_cublas.data().get(), ldC);
    }
    double cublas_time = timer.seconds() / 100.0;
    printf("  cuBLAS Time: %.4f ms, %.1f GFlop/s\n", cublas_time * 1000, gflops / cublas_time);

    // --- Check ---
    thrust::host_vector<TC> h_wgmma  = d_C_wgmma;
    thrust::host_vector<TC> h_cublas = d_C_cublas;
    float max_error = 0.0f;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * ldC + j; // 使用对齐的 ldC 进行遍历验证
            float diff = fabs(float(h_wgmma[idx]) - float(h_cublas[idx]));
            max_error = fmax(max_error, diff);
        }
    }
    printf("  Correctness: %s (Max Error: %.4f)\n", max_error < 1.0f ? "PASSED" : "FAILED", max_error);

    cublasDestroy(handle);
#endif
    return 0;
}