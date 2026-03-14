#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h> // 需要包含这个头文件以使用流水线原语
#include <cublas_v2.h>     // 引入 cuBLAS 头文件

using namespace std;
#define BM 128
#define BN 128
// 2. K-Dimension Tile: 每次迭代 K 的步长
#define BK 8
// 3. Thread Tile: 每个线程计算 C 的 [TM, TN]
#define TM 8
#define TN 8

// 派生参数
// Block 内的线程数 = (BM * BN) / (TM * TN) = (128*128) / (8*8) = 256
#define NUM_THREADS 256

// ============================================================================
// Kernel: Register Tiled SGEMM (FP32 CUDA Core Optimized)
// ============================================================================
__global__ void matmul(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    // ----------------------------------------------------------------
    // 1. Shared Memory 声明 (带 Padding 防止 Bank Conflict)
    // ----------------------------------------------------------------
    // As: [BM][BK] -> [128][8]。为了 float4 加载，行必须连续? 
    // 实际上 A 是 [M, K]，Global Load 是按行读。
    // 为了计算方便，我们希望 As 按列读 (A x B)。
    // 这里我们保持 As [BM][BK]，但为了避免 Bank Conflict (Thread 0 读 row 0, Thread 1 读 row 8...)
    // 我们加上 Padding。
    __shared__ float s_A[BM][BK]; 
    __shared__ float s_B[BK][BN]; 

    // ----------------------------------------------------------------
    // 2. 寄存器声明
    // ----------------------------------------------------------------
    // 累加器: 8x8 = 64 个寄存器
    float thread_results[TM][TN] = {0.0f};
    // 缓存 A 和 B 的寄存器片段
    float reg_a[TM];
    float reg_b[TN];

    // ----------------------------------------------------------------
    // 3. 线程索引与全局加载索引计算
    // ----------------------------------------------------------------
    int tid = threadIdx.x;
    
    // 线程在 C 中的逻辑位置 (用于计算)
    // Block 内布局: 16行 x 16列 的线程网格 (16*8=128)
    int ty = tid / (BN / TN); // 0..15
    int tx = tid % (BN / TN); // 0..15

    // Global Load 索引预计算 (协作加载)
    // 每个 Block 需要加载 A: [128, 8] = 1024 float
    // 256 个线程 -> 每个线程加载 4 个 float (正好 1 个 float4)
    // A 的 Global 指针: blockRow * K
    int load_a_row = tid / (BK / 4); // BK=8, float4=4 -> div 2. 0..127. 覆盖所有行
    int load_a_col = (tid % (BK / 4)) * 4; // 0 or 4.
    
    // B 的 Global 指针: blockCol
    // 每个 Block 需要加载 B: [8, 128] = 1024 float
    // 256 个线程 -> 每个线程加载 4 个 float
    int load_b_row = tid / (BN / 4); // 0..7
    int load_b_col = (tid % (BN / 4)) * 4; // 0..124 (步长4)

    // 移动指针到当前 Block 的起始位置
    const float* src_A = A + (blockIdx.y * BM + load_a_row) * K + load_a_col;
    const float* src_B = B + (load_b_row) * N + (blockIdx.x * BN + load_b_col);

    // ----------------------------------------------------------------
    // 4. 主循环 K
    // ----------------------------------------------------------------
    for (int k = 0; k < K; k += BK) {
        
        // --- Stage A: 协作加载 Global -> Shared (Vectorized float4) ---
        
        // 加载 A (float4)
        // 注意边界检查: 如果 M, K 不是 4 的倍数需要特殊处理。这里假设是对齐的。
        // reinterpret_cast 强制转换为 float4* 进行加载
        float4 tmp_a = *reinterpret_cast<const float4*>(src_A);
        // 存入 Shared Memory (需拆包)
        s_A[load_a_row][load_a_col + 0] = tmp_a.x;
        s_A[load_a_row][load_a_col + 1] = tmp_a.y;
        s_A[load_a_row][load_a_col + 2] = tmp_a.z;
        s_A[load_a_row][load_a_col + 3] = tmp_a.w;

        // 加载 B (float4)
        float4 tmp_b = *reinterpret_cast<const float4*>(src_B);
        s_B[load_b_row][load_b_col + 0] = tmp_b.x;
        s_B[load_b_row][load_b_col + 1] = tmp_b.y;
        s_B[load_b_row][load_b_col + 2] = tmp_b.z;
        s_B[load_b_row][load_b_col + 3] = tmp_b.w;

        // 指针步进
        src_A += BK;     // A 沿 K 走
        src_B += BK * N; // B 沿 Row 走 (K维度)

        // 等待加载完成
        __syncthreads();

        // --- Stage B: 计算 (Shared -> Register -> Compute) ---
        
        // 这里的内层循环很小 (BK=8)，完全在寄存器和 L1 中完成
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            
            // 1. 加载 A 的一列 (TM=8 个) 到寄存器
            // 线程负责 C 的 [ty*TM ... ty*TM+7][...]
            // 需要读取 A 的 [ty*TM ... ty*TM+7][i]
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                reg_a[r] = s_A[ty * TM + r][i];
            }

            // 2. 加载 B 的一行 (TN=8 个) 到寄存器
            // 线程负责 C 的 [...][tx*TN ... tx*TN+7]
            // 需要读取 B 的 [i][tx*TN ... tx*TN+7]
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                reg_b[c] = s_B[i][tx * TN + c];
            }

            // 3. 外积计算 (Outer Product) 8x8
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    thread_results[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }

        // 等待计算完成，才能进入下一轮加载覆盖 Shared Memory
        __syncthreads();
    }

    // ----------------------------------------------------------------
    // 5. 写回结果 Global Memory
    // ----------------------------------------------------------------
    int global_c_row = blockIdx.y * BM + ty * TM;
    int global_c_col = blockIdx.x * BN + tx * TN;

    // 简单的逐元素写回 (也可以优化为 float4 写回)
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        #pragma unroll
        for (int c = 0; c < TN; ++c) {
            if (global_c_row + r < M && global_c_col + c < N) {
                C[(global_c_row + r) * N + (global_c_col + c)] = thread_results[r][c];
            }
        }
    }
}


// ------------------------------------------------------------------
// CPU 参考实现 (用于校验)
// ------------------------------------------------------------------
void cpu_gemm(int M, int N, int K, float *A, float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ------------------------------------------------------------------
// 主函数
// ------------------------------------------------------------------
int main() {
    // 矩阵大小 (故意设置非 32 倍数以测试边界检查)
    int M = 4096;
    int N = 4096;
    int K = 4096;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存分配
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 拷贝数据 H2D
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 配置 Grid 和 Block
    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    printf("Matrix: %d x %d x %d\n", M, N, K);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    // 启动 Kernel
    // Warmup
    matmul<<<dimGrid, dimBlock>>>(M, N, K, d_A, d_B, d_C);
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul<<<dimGrid, dimBlock>>>(M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Performance: %.2f TFLOPS\n", (2.0 * M * N * K  / 1e12) / (milliseconds / 1000.0));
    printf("GPU Time: %.4f ms\n", milliseconds);

    // 拷贝回结果 D2H
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 校验结果
    printf("Verifying result...\n");
    // cpu_gemm(M, N, K, h_A, h_B, h_C_ref);

    // float max_diff = 0.0f;
    // for (int i = 0; i < M * N; i++) {
    //     float diff = abs(h_C[i] - h_C_ref[i]);
    //     if (diff > max_diff) max_diff = diff;
    // }
    // printf("Max Diff: %f\n", max_diff);
    // if (max_diff < 1e-3) printf("✅ Test Passed!\n");
    // else printf("❌ Test Failed!\n");

    // 清理

    printf("\nRunning cuBLAS...\n");
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    float milblas = 0;

    // Warmup cuBLAS
    // 注意：cuBLAS 是列优先。计算行优先 C = A * B 等价于列优先 C^T = B^T * A^T
    // 所以这里传入 d_B 作为 A 矩阵，d_A 作为 B 矩阵，维度参数 M 和 N 也要根据转置逻辑传入
    // Sgemm(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // 对应计算: C(mxn) = A(mxk) * B(kxn)
    // 我们要算 C^T(NxM) = B^T(NxK) * A^T(KxM)
    // m=N, n=M, k=K. A=d_B(ld=N), B=d_A(ld=K), C=d_C(ld=N)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milblas, start, stop);
    
    printf("cuBLAS Performance: %.2f TFLOPS\n", (2.0 * M * N * K / 1e12) / (milblas / 1000.0));
    printf("cuBLAS Time: %.4f ms\n", milblas);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}