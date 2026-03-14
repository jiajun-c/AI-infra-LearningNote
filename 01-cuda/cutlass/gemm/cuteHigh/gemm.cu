// 保存为 cute_gemm_demo.cu
// 编译命令: nvcc cute_gemm_demo.cu -o cute_gemm -I /path/to/cutlass/include -arch=sm_80 --expt-relaxed-constexpr

#include "cute/container/array_subbyte.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer_base.hpp"
#include "cute/pointer_sparse.hpp"
#include "cute/pointer_swizzle.hpp"
#include "cute/swizzle.hpp"
#include "cute/util/print.hpp"
#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

// =================================================================================
// 核心 Kernel：在 Device 上运行
// =================================================================================
template <class ProblemShape, class CtaTiler,
          class TA_ptr, class AStride, class ASmemLayout, class AThreadLayout,
          class TB_ptr, class BStride, class BSmemLayout, class BThreadLayout,
          class TC_ptr, class CStride, class CThreadLayout>
__global__ void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA_ptr A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB_ptr B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC_ptr C, CStride dC, CThreadLayout tC)
{
    // 1. 【全量张量】将原始指针包装成 Tensor
    // 代表显存中完整的矩阵 (M, K) 和 (N, K)
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); 
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); 
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); 

    // 2. 【CTA Tiling】切出一块给当前 Thread Block
    // blockIdx.x, blockIdx.y 决定了当前 Block 处理哪一块
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N, BLK_K, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M, BLK_N)
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        print(gA);print("\n");
        print(gB);print("\n");
        print(gC);print("\n");


    }
    // 3. 【Shared Memory】定义共享内存张量
    // cosize_v 会自动计算 Layout 需要多大的物理内存
    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            

    // 4. 【Thread Partitioning - Load】数据搬运视角
    // 决定当前线程负责搬运 gA/gB 中的哪些数据到 sA/sB
    // tA/tB 定义了线程如何覆盖数据块 (32x8 的线程排布)
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        print(gA);print(" ga\n");
        print(tA);print("\n");
        print(sA);print("\n");
    }                  
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  

    // 5. 【Thread Partitioning - Compute】计算视角
    // 决定当前线程负责计算 sA/sB 中的哪些数据，并写入寄存器 rC
    // tC 定义了计算时的线程排布 (16x16)
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});  
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        print("sA\n");print(sA);print("\n");
        print(tC);print("\n");
        print(tCsA);print("\n");

    }
    // print(tCsA);print("\n");
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});  

 
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   

    // 定义寄存器累加器 (Accumulators)，结构同 C 的分块
    Tensor tCrC = make_tensor_like(tCgC); 
    clear(tCrC); // 初始化为 0

    // 6. 【Main Loop】K 维度循环
    auto K_TILE_MAX = size<2>(tAgA); // 计算 K 维度要循环多少次

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // [Copy]: Global -> Shared
        // CuTe 的 copy 会自动根据 Layout 生成最优的向量化加载指令 (LDG.128 等)
        copy(tAgA(_,_,k_tile), tAsA);
        copy(tBgB(_,_,k_tile), tBsB);

        // 简单的同步，等待拷贝完成 (在 Hopper 架构可用 TMA 异步拷贝优化，这里用最基础的)
        __syncthreads(); 

        // [Compute]: Shared -> Register
        // CuTe 的 gemm 会计算该 tile 的乘积并累加到 tCrC
        gemm(tCsA, tCsB, tCrC);

        // 等待计算完成，防止下次 Copy 覆盖了还没算完的 Shared Memory
        __syncthreads(); 
    }

    // 7. 【Epilogue】寄存器 -> Global
    // 将计算结果写回全局显存
    copy(tCrC, tCgC);
}

// =================================================================================
// Host 端 Setup：定义 Layout 并启动 Kernel
// =================================================================================
void run_gemm_demo(int m, int n, int k, float* d_A, float* d_B, float* d_C) {
    // 1. 定义问题形状
    auto prob_shape = make_shape(m, n, k);

    // 2. 定义 Global Memory 的 Layout (Strides)
    // 假设 A 是 Column-Major (M, K) -> Stride (1, M)
    // 假设 B 是 Column-Major (N, K) -> Stride (1, N) (这里对应 NT 模式)
    auto dA = make_stride(Int<1>{}, m); 
    auto dB = make_stride(Int<1>{}, n); 
    auto dC = make_stride(Int<1>{}, m); 

    // 3. 定义 Tiling (分块) 大小
    // 每个 Block 处理 128x128x8 的数据
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // 4. 定义 Shared Memory Layout
    // 只是简单的一维连续排布，但在物理上是 (128, 8)
    auto sA = make_layout(make_shape(bM, bK)); 
    auto sB = make_layout(make_shape(bN, bK)); 
    // sC 不需要显式定义，因为我们直接从寄存器写回 Global

    // 5. 定义 Thread Layout (线程排布)
    // 这里的 (32, 8) 表示 256 个线程排成 32行8列 来搬运数据
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{})); 
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{})); 
    // 计算时的线程排布：16x16 = 256 个线程
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); 

    // 计算 Grid 大小
    dim3 dimBlock(size(tC)); // 256 threads
    dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));

    std::cout << "Launching Kernel with Grid: (" << dimGrid.x << ", " << dimGrid.y << ")" << std::endl;

    // 启动 Kernel
    gemm_device<<<dimGrid, dimBlock>>>(
        prob_shape, cta_tiler,
        d_A, dA, sA, tA,
        d_B, dB, sB, tB,
        d_C, dC, tC // sC 其实没用到，占位
    );
}

int main() {
    int m = 1024;
    int n = 1024;
    int k = 1024;

    std::cout << "Running CuTe GEMM Demo (M=" << m << ", N=" << n << ", K=" << k << ")..." << std::endl;

    // Host 内存分配与初始化
    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(n * k, 1.0f); // 全 1，结果应该是 k (1024)
    std::vector<float> h_C(m * n, 0.0f);

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, n * k * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Copy H2D
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));

    // 运行
    run_gemm_demo(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Copy D2H
    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    // C = A * B^T. A, B 全是 1. C 的每个元素应该是 K (1024.0)
    std::cout << "Validation: C[0] = " << h_C[0] << " (Expected: 1024.0)" << std::endl;
    std::cout << "Validation: C[end] = " << h_C[m*n-1] << " (Expected: 1024.0)" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}