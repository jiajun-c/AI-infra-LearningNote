#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include <cute/tensor.hpp>
#include <vector>
#include <iostream>

using namespace cute;

// 定义矩阵维度
const int M = 128;
const int N = 256;
const int K = 512;

template <class ProblemShape, class CtaTiler,
          class TA_ptr, class AStride, class ASmemLayout, class AThreadLayout,
          class TB_ptr, class BStride, class BSmemLayout, class BThreadLayout,
          class TC_ptr, class CStride, class CThreadLayout>
__global__ void
gemm_device(ProblemShape shape, CtaTiler cta_tiler,
            TA_ptr A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB_ptr B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC_ptr C, CStride dC, CThreadLayout tC
        ) {
    // 【修正 1】必须传入 Stride (dA, dB, dC)，否则默认是列主序，读不到正确数据
    auto mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape), dA);
    auto mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape), dB);
    auto mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape), dC);

    auto coord_idx =  make_coord(blockIdx.x, blockIdx.y, _);
    
    // 切片 (Tiling)
    // gA: (BLK_M, BLK_K, k)
    auto gA = local_tile(mA, cta_tiler, coord_idx, Step<_1, X, _1>{});
    // gB: (BLK_N, BLK_K, k)
    auto gB = local_tile(mB, cta_tiler, coord_idx, Step<X, _1, _1>{});
    // gC: (BLK_M, BLK_N)
    auto gC = local_tile(mC, cta_tiler, coord_idx, Step<_1, _1, X>{});
    
    // Shared Memory 定义
    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];
    
    auto sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // 线程搬运分工 (Partitioning for Copy)
    auto tAgA = local_partition(gA, tA, threadIdx.x);
    auto tAsA = local_partition(sA, tA, threadIdx.x);

    auto tBgB = local_partition(gB, tB, threadIdx.x);
    auto tBsB = local_partition(sB, tB, threadIdx.x);
    
    // 线程计算结果分工 (Partitioning for Compute)
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

    int K_TILES = size<2>(tAgA);
    float acc = 0;
    int tid = threadIdx.x;
    
    // 解析当前线程在 Tile 中的逻辑坐标 (row, col)
    // tC 是 RowMajor (32, 32)，所以 tid = row * 32 + col
    int rowInTile = tid / 32;
    int colInTile = tid % 32;

    for (int i = 0; i < K_TILES; i++) {
        // 1. Load Data
        copy(tAgA(_, _, i), tAsA);
        copy(tBgB(_, _, i), tBsB);

        // 同步 1：确保所有线程都 Load 完了数据
        __syncthreads();
        
        // 2. Compute
        // 矩阵乘法: C(row, col) += sum( A(row, k) * B(col, k) )
        // 注意：这里 sB 的访问取决于 sB 的 Layout 和 mB 的逻辑定义
        // 下文 Host 代码中我们将 B 定义为 (N, K)，所以 sB(col, k) 是正确的
        for (int k = 0; k < size<1>(sA); k++) {
            acc += sA(rowInTile, k) * sB(colInTile, k); 
        }

        // 【修正 2】同步 2：至关重要！
        // 必须确保所有线程算完了当前块，才能进入下一次循环覆盖 Shared Memory
        __syncthreads();
    }
    
    // 3. Store Result
    // 注意：这里用 copy 直接覆盖。如果 C 需要累加，应该先 load 再 add。
    // 但鉴于 host 做了 memset 0，这里直接写是安全的。
    copy(make_tensor(&acc, make_layout(Int<1>{})), tCgC);
}

void gemm(std::vector<float>&h_A, std::vector<float>&h_B, std::vector<float>& h_C) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A.size()*sizeof(float));
    cudaMalloc(&d_B, h_B.size()*sizeof(float));
    cudaMalloc(&d_C, h_C.size()*sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, h_C.size()*sizeof(float));

    // 【修正 3】Stride 定义必须匹配 Row-Major 物理内存
    // A (M, K) RowMajor -> Stride (K, 1)
    auto dA = make_stride(K, Int<1>{});
    
    // B (N, K) Logical -> 物理是 K*N RowMajor
    // 在 CuTe 中，我们通常把 B 视为 (N, K) 形状。
    // 如果物理上 B 是标准的 K行N列 RowMajor 矩阵，访问 B[k][n] (地址 k*N + n)。
    // 映射到 Logical B(n, k)：地址应该是 k*N + n*1。
    // 所以 Stride 应该是 (1, N)。
    // 注：你原本写的 make_stride(N, 1) 对应的是 Column Major 或者 B是 NxK 物理矩阵。
    // 这里假设 B 是标准的 KxN 矩阵。
    auto dB = make_stride(Int<1>{}, N); 

    // C (M, N) RowMajor -> Stride (N, 1)
    auto dC = make_stride(N, Int<1>{});
    
    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto bK = Int<32>{};

    auto cta_tiler = make_shape(bM, bN, bK);
    
    // Shared Memory Layouts
    // sA: (bM, bK) -> (32, 32)
    auto sALayout = make_layout(make_shape(bM, bK), GenRowMajor{});
    // sB: (bN, bK) -> (32, 32)
    // 注意：Global B 切片出来是 (BLK_N, BLK_K)。Shared B 形状必须匹配，否则 Copy 会错位。
    // 原代码写的 make_shape(bK, bN) 是 (32, 32)，但是维度含义反了。
    auto sBLayout = make_layout(make_shape(bN, bK), GenRowMajor{});

    // Thread Layouts (32x32 = 1024 threads)
    auto tA = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
    auto tB = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
    auto tC = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});

    dim3 dimBlock(1024);
    dim3 dimGrid((M + 32 - 1) / 32, (N + 32 - 1) / 32);

    auto shape_MNK = make_shape(M, N, K);

    gemm_device<<<dimGrid, dimBlock>>>(
        shape_MNK, cta_tiler,
        d_A, dA, sALayout, tA,
        d_B, dB, sBLayout, tB,
        d_C, dC, tC
    );
    
    cudaMemcpy(h_C.data(), d_C, h_C.size()*sizeof(float), cudaMemcpyDeviceToHost);
    
    // 简单的 CPU 验证
    float ref = 0;
    for(int k=0; k<K; ++k) ref += h_A[0*K + k] * h_B[k*N + 0]; // C[0,0]
    printf("GPU: %f, CPU Ref: %f\n", h_C[0], ref);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    // 初始化数据
    std::vector<float>h_A(M*K);
    std::vector<float>h_B(N*K); // K行N列
    std::vector<float> h_C(M * N, 0.0f);
    
    for(int i=0; i<h_A.size(); ++i) h_A[i] = (float)(rand() % 10) / 10.0f;
    for(int i=0; i<h_B.size(); ++i) h_B[i] = (float)(rand() % 10) / 10.0f;

    gemm(h_A, h_B, h_C);
    // printf("%f %f\n", h_C[0], h_C[1024]);
}