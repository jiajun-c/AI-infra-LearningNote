#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

// =========================================================
// [修复关键点] 引入更全的 SM90 头文件
// =========================================================
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/mma_sm90_gmma.hpp> // 修复 wgmma_wait_group undefined 的一种可能

using namespace cute;

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess) {                                           \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tick() { cudaEventRecord(start); }
    float tock() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// =================================================================================
// Kernel 1: SM80 (Ampere) - 全 FP16
// =================================================================================
template <class ProblemShape, class CtaTiler, class TA, class TB, class TC>
__global__ void gemm_sm80_fp16_kernel(TA* d_A, TB* d_B, TC* d_C, 
                                      ProblemShape shape_MNK, CtaTiler cta_tiler) {
    auto gA = make_tensor(make_gmem_ptr(d_A), select<0, 2>(shape_MNK), make_stride(size<2>(shape_MNK), Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(d_B), select<1, 2>(shape_MNK), make_stride(size<2>(shape_MNK), Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(d_C), select<0, 1>(shape_MNK), make_stride(size<1>(shape_MNK), Int<1>{}));

    extern __shared__ TA smem[];
    TA* sA_ptr = smem;
    TB* sB_ptr = smem + cosize(get<0>(cta_tiler)) * cosize(get<2>(cta_tiler));

    auto sA = make_tensor(make_smem_ptr(sA_ptr), make_layout(make_shape (get<0>(cta_tiler), get<2>(cta_tiler)), LayoutRight{}));
    auto sB = make_tensor(make_smem_ptr(sB_ptr), make_layout(make_shape (get<1>(cta_tiler), get<2>(cta_tiler)), LayoutRight{}));

    int blk_idx_x = blockIdx.x;
    int blk_idx_y = blockIdx.y;
    auto cta_coord = make_coord(blk_idx_x, blk_idx_y, _);
    auto gA_tile = local_tile(gA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    auto gB_tile = local_tile(gB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    auto gC_tile = local_tile(gC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // 保持使用 SM80 指令集
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN; 
    
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;
    
    auto tiled_mma = make_tiled_mma(MMA_Atom{}, make_layout(Shape<_2, _2, _1>{})); 
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr_mma.partition_fragment_C(gC_tile);
    clear(tCrC);

    auto loader = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{}, make_layout(Shape<_32, _4>{}, LayoutRight{}));
    auto thr_loader = loader.get_thread_slice(threadIdx.x);

    int K_tiles = size<2>(shape_MNK) / size<2>(cta_tiler);
    for (int k = 0; k < K_tiles; ++k) {
        auto tLgA = thr_loader.partition_S(gA_tile(_, _, k));
        auto tLsA = thr_loader.partition_D(sA);
        auto tLgB = thr_loader.partition_S(gB_tile(_, _, k));
        auto tLsB = thr_loader.partition_D(sB);

        copy(loader, tLgA, tLsA);
        copy(loader, tLgB, tLsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        auto tCsA = thr_mma.partition_fragment_A(sA);
        auto tCsB = thr_mma.partition_fragment_B(sB);
        auto tCrA = thr_mma.make_fragment_A(tCsA);
        auto tCrB = thr_mma.make_fragment_B(tCsB);
        copy(tCsA, tCrA); 
        copy(tCsB, tCrB); 
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        __syncthreads();
    }
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) d_C[0] = tCrC(0);
}

// =================================================================================
// Kernel 2: SM90 (Hopper) - 全 FP16
// =================================================================================
template <class ProblemShape, class CtaTiler, class TA, class TB, class TC>
__global__ void gemm_sm90_fp16_kernel(TA* d_A, TB* d_B, TC* d_C, 
                                      ProblemShape shape_MNK, CtaTiler cta_tiler) {
    auto gA = make_tensor(make_gmem_ptr(d_A), select<0, 2>(shape_MNK), make_stride(size<2>(shape_MNK), Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(d_B), select<1, 2>(shape_MNK), make_stride(size<2>(shape_MNK), Int<1>{}));

    extern __shared__ TA smem[];
    TA* sA_ptr = smem;
    TB* sB_ptr = smem + cosize(get<0>(cta_tiler)) * cosize(get<2>(cta_tiler));

    auto sA_layout = composition(Swizzle<3,3,3>{}, make_layout(make_shape (get<0>(cta_tiler), get<2>(cta_tiler)), LayoutRight{}));
    auto sB_layout = composition(Swizzle<3,3,3>{}, make_layout(make_shape (get<1>(cta_tiler), get<2>(cta_tiler)), LayoutRight{}));
    auto sA = make_tensor(make_smem_ptr(sA_ptr), sA_layout);
    auto sB = make_tensor(make_smem_ptr(sB_ptr), sB_layout);

    // =========================================================
    // [修复关键点 1] F16_SS 是模板，需要 <K, K>
    // =========================================================
    using MMA_Op = SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>; 
    
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;

    auto tiled_mma = make_tiled_mma(MMA_Atom{}, make_layout(Shape<_2, _2, _1>{})); 
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr_mma.partition_fragment_C(sA); 
    clear(tCrC);

    auto loader = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{}, make_layout(Shape<_32, _4>{}, LayoutRight{}));
    auto thr_loader = loader.get_thread_slice(threadIdx.x);

    int blk_idx_x = blockIdx.x;
    int blk_idx_y = blockIdx.y;
    auto cta_coord = make_coord(blk_idx_x, blk_idx_y, _);
    auto gA_tile = local_tile(gA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    auto gB_tile = local_tile(gB, cta_tiler, cta_coord, Step<X, _1, _1>{});

    int K_tiles = size<2>(shape_MNK) / size<2>(cta_tiler);
    for (int k = 0; k < K_tiles; ++k) {
        auto tLgA = thr_loader.partition_S(gA_tile(_, _, k));
        auto tLsA = thr_loader.partition_D(sA);
        auto tLgB = thr_loader.partition_S(gB_tile(_, _, k));
        auto tLsB = thr_loader.partition_D(sB);

        copy(loader, tLgA, tLsA);
        copy(loader, tLgB, tLsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        auto tCsA = thr_mma.partition_fragment_A(sA);
        auto tCsB = thr_mma.partition_fragment_B(sB);
        
        gemm(tiled_mma, tCrC, tCsA, tCsB, tCrC);
        
        // =========================================================
        // [修复关键点 2] 如果依然 undefined，请尝试 cute::wgmma_wait_group<0>()
        // =========================================================
        cute::warpgroup_wait<0>();
        __syncthreads();
    }
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) d_C[0] = tCrC(0);
}

int main() {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    
    using TA = half;
    using TB = half;
    using TC = half;

    TA *d_A, *d_B;
    TC *d_C;
    size_t size_A = M * K * sizeof(TA);
    size_t size_B = N * K * sizeof(TB);
    size_t size_C = M * N * sizeof(TC);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemset(d_A, 1, size_A));
    CUDA_CHECK(cudaMemset(d_B, 1, size_B));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    int n_iter = 100;
    GpuTimer timer;

    {
        auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<32>{});
        auto shape_MNK = make_shape(M, N, K);
        
        dim3 dimBlock(128); 
        dim3 dimGrid(M / 128, N / 128);
        int smem_size = (128*32 + 128*32) * sizeof(TA);

        gemm_sm80_fp16_kernel<<<dimGrid, dimBlock, smem_size>>>(d_A, d_B, d_C, shape_MNK, cta_tiler);
        
        timer.tick();
        for(int i=0; i<n_iter; ++i) {
            gemm_sm80_fp16_kernel<<<dimGrid, dimBlock, smem_size>>>(d_A, d_B, d_C, shape_MNK, cta_tiler);
        }
        float ms = timer.tock();
        
        double tflops = (2.0 * M * N * K * n_iter) / (ms * 1e9);
        std::cout << "[SM80 MMA  FP16] Avg Time: " << ms / n_iter << " ms | Perf: " << tflops << " TFLOPS" << std::endl;
    }

    {
        auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<64>{}); 
        auto shape_MNK = make_shape(M, N, K);

        dim3 dimBlock(128); 
        dim3 dimGrid(M / 128, N / 128);
        int smem_size = (128*64 + 128*64) * sizeof(TA);

        cudaFuncSetAttribute(gemm_sm90_fp16_kernel<decltype(shape_MNK), decltype(cta_tiler), TA, TB, TC>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        gemm_sm90_fp16_kernel<<<dimGrid, dimBlock, smem_size>>>(d_A, d_B, d_C, shape_MNK, cta_tiler);

        timer.tick();
        for(int i=0; i<n_iter; ++i) {
            gemm_sm90_fp16_kernel<<<dimGrid, dimBlock, smem_size>>>(d_A, d_B, d_C, shape_MNK, cta_tiler);
        }
        float ms = timer.tock();

        double tflops = (2.0 * M * N * K * n_iter) / (ms * 1e9);
        std::cout << "[SM90 WGMMA FP16] Avg Time: " << ms / n_iter << " ms | Perf: " << tflops << " TFLOPS" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}