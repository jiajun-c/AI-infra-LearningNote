#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

// =================================================================================
// Kernel
// =================================================================================
template<class ProblemShape, class CtaTiler,
        class TA_ptr, class AStride, class ASmemLayout, class AThreadLayout,
        class TB_ptr, class BStride, class BSmemLayout, class BThreadLayout,
        class TC_ptr, class CStride, class CThreadLayout>
__global__ void gemm(ProblemShape shape, CtaTiler cta_tiler,
                    TA_ptr A, AStride dA, ASmemLayout sALayout, AThreadLayout tA,
                    TB_ptr B, BStride dB, BSmemLayout sBLayout, BThreadLayout tB,
                    TC_ptr C, CStride dC, CThreadLayout tC) {
    // 1. Global Tensor
    // A: (M, K)
    auto mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape), dA); 
    // B: (N, K). 注意：这里我们逻辑上认为 B 是 (N, K)，stride 会处理物理映射
    auto mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape), dB); 
    // C: (M, N)
    auto mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape), dC); 

    auto cood_idx = make_coord(blockIdx.x, blockIdx.y, _);
    
    // 2. Local Tile
    auto gA = local_tile(mA, cta_tiler, cood_idx, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)
    // gB 切分出来的形状是 (BLK_N, BLK_K, k)
    auto gB = local_tile(mB, cta_tiler, cood_idx, Step<X, _1, _1>{}); 
    auto gC = local_tile(mC, cta_tiler, cood_idx, Step<_1, _1, X>{}); 

    // 3. Shared Memory
    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];
    
    auto sA = make_tensor(make_smem_ptr(smemA), sALayout);
    auto sB = make_tensor(make_smem_ptr(smemB), sBLayout);

    // 4. Partitioning
    auto tAgA = local_partition(gA, tA, threadIdx.x);
    auto tAsA = local_partition(sA, tA, threadIdx.x);
    
    auto tBgB = local_partition(gB, tB, threadIdx.x);
    auto tBsB = local_partition(sB, tB, threadIdx.x);

    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    
    float tCrC = 0.0f; 

    // 5. Main Loop
    auto K_TILE_MAX = size<2>(gA);
    for (int k = 0; k < K_TILE_MAX; k++) {
        // Copy Global -> Shared
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);
        
        __syncthreads();
        
        // Compute
        int tid = threadIdx.x;
        // tC 是 ColMajor (32, 32) -> tid 0=(0,0), tid 1=(1,0) ...
        int row_in_block = tid % 32; 
        int col_in_block = tid / 32;
        
        // 循环 K (inner dim)
        for (int i = 0; i < size<1>(sA); i++) {
            // A(m, k) * B(n, k)
            // sA: (32, 32) ColMajor -> sA(row, i)
            // sB: (32, 32) RowMajor -> sB(col, i)  <-- 这里 i 是 K 维度
            tCrC += sA(row_in_block, i) * sB(col_in_block, i);
        }
        
        __syncthreads();
    }
    
    copy(make_tensor(&tCrC, make_layout(Int<1>{})), tCgC);
}

// =================================================================================
// CPU 参考实现
// =================================================================================
// 标准矩阵乘法 C = A * B
// A: M*K ColMajor
// B: K*N ColMajor
void cpu_gemm(int M, int N, int K, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < M; ++i) { // Row
        for (int j = 0; j < N; ++j) { // Col
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i, k] -> i + k*M
                // B[k, j] -> k + j*K
                sum += A[i + k * M] * B[k + j * K];
            }
            C[i + j * M] = sum;
        }
    }
}

int main() {
    int M = 128;
    int N = 128;
    int K = 128;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N); // K*N 物理大小
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    // 初始化
    for(int i=0; i<h_A.size(); ++i) h_A[i] = static_cast<float>(rand())/RAND_MAX;
    for(int i=0; i<h_B.size(); ++i) h_B[i] = static_cast<float>(rand())/RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A.size()*sizeof(float));
    cudaMalloc(&d_B, h_B.size()*sizeof(float));
    cudaMalloc(&d_C, h_C.size()*sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, h_C.size()*sizeof(float));

    // --- 关键修改 1: Global Memory Layout ---
    // A: (M, K) ColMajor -> Stride (1, M)
    auto dA = make_stride(Int<1>{}, M);
    
    // B: 物理上是 (K, N) ColMajor -> 访问 B[k, n] 是 k + n*K
    // 我们将其视为逻辑上的 (N, K) RowMajor -> 访问 B'[n, k] 是 n*K + k
    // 这两个公式等价。所以我们定义 stride 为 (K, 1)
    // Shape (N, K) -> Stride (K, 1)
    auto dB = make_stride(K, Int<1>{});

    // C: (M, N) ColMajor -> Stride (1, M)
    auto dC = make_stride(Int<1>{}, M);
    
    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // --- 关键修改 2: Shared Memory Layout ---
    // sA: (M, K) ColMajor. 没问题。
    auto sALayout = make_layout(make_shape(bM, bK), GenColMajor{});
    
    // sB: (N, K) RowMajor. 
    // 为什么用 RowMajor? 
    // 因为在 Global 加载时，K 维度（Stride 1）是连续的。
    // 我们希望 Shared Memory 写入也是连续的（Stride 1）。
    auto sBLayout = make_layout(make_shape(bN, bK), GenRowMajor{});

    // --- 关键修改 3: Thread Layout ---
    auto tA = make_layout(make_shape(Int<32>{}, Int<32>{}), GenColMajor{});
    
    // tB: RowMajor (32, 32).
    // Mapping: tid -> (n, k) = (tid/32, tid%32)
    // 当 tid + 1 时，k + 1。
    // Global Access Addr (RowMajor view): n*K + k. 
    // k + 1 意味着 Global 地址连续 -> 合并访问 (Coalesced Load)。✅
    // Smem Access Addr (RowMajor): n*K + k.
    // k + 1 意味着 Smem 地址连续 -> 无 Bank Conflict。✅
    auto tB = make_layout(make_shape(Int<32>{}, Int<32>{}), GenRowMajor{});
    
    auto tC = make_layout(make_shape(Int<32>{}, Int<32>{}), GenColMajor{});

    dim3 dimBlock(1024);
    dim3 dimGrid((M + 32 - 1) / 32, (N + 32 - 1) / 32);

    auto shape_MNK = make_shape(M, N, K);
    
    gemm<<<dimGrid, dimBlock>>>(
        shape_MNK, cta_tiler,
        d_A, dA, sALayout, tA,
        d_B, dB, sBLayout, tB,
        d_C, dC, tC
    );

    cudaMemcpy(h_C.data(), d_C, h_C.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cpu_gemm(M, N, K, h_A, h_B, h_C_ref);

    float max_diff = 0.0f;
    for (int i = 0; i < h_C.size(); ++i) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max Error: " << max_diff << std::endl;
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}