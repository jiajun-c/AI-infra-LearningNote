#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h> // 需要包含这个头文件以使用流水线原语
#include <cublas_v2.h>     // 引入 cuBLAS 头文件
#include <mma.h>           // 引入 Tensor Core WMMA API

using namespace std;
using namespace nvcuda;

// ------------------------------------------------------------------
// 参数配置
// ------------------------------------------------------------------
// WMMA 处理的 Tile 大小
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8 

// ------------------------------------------------------------------
// Kernel: 使用 Tensor Core (TF32) 实现 SGEMM
// 每个 Warp 计算 C 的一个 16x16 块
// ------------------------------------------------------------------
__global__ void matmul(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    // 声明 WMMA 片段 (Fragments)
    // 使用 TF32 精度 (Ampere/Hopper 特性): float 输入 -> float 累加
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 计算当前 Warp 负责的 C 矩阵的坐标 (Row, Col)
    // 假设 BlockDim.x 是 32 (1个Warp) 或 32的倍数
    int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int numWarpsX = (gridDim.x * blockDim.x) / 32;
    
    // 我们将一维的 globalWarpId 映射到二维的 Tile 坐标
    // 假设 Grid 覆盖了 (N / 16) * (M / 16) 个 Tiles
    int tiles_per_row = N / WMMA_N;
    
    int tileRow = globalWarpId / tiles_per_row;
    int tileCol = globalWarpId % tiles_per_row;

    // 计算实际的像素坐标
    int cRow = tileRow * WMMA_M;
    int cCol = tileCol * WMMA_N;

    // 边界检查
    if (cRow >= M || cCol >= N) return;

    // 主循环：沿着 K 维度迭代
    // 每次处理 WMMA_K (8) 的深度
    for (int i = 0; i < K; i += WMMA_K) {
        // 1. 加载 A 的片段 (16x8)
        // A 是 Row Major: [M, K]
        // 指针指向 A[cRow, i]
        const float* ptrA = A + cRow * K + i;
        wmma::load_matrix_sync(a_frag, ptrA, K);

        // 2. 加载 B 的片段 (8x16)
        // B 是 Row Major: [K, N]
        // 指针指向 B[i, cCol]
        const float* ptrB = B + i * N + cCol;
        wmma::load_matrix_sync(b_frag, ptrB, N);

        // 3. 执行矩阵乘累加: D = A * B + C
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. 写回结果
    // 指针指向 C[cRow, cCol]
    float* ptrC = C + cRow * N + cCol;
    wmma::store_matrix_sync(ptrC, c_frag, N, wmma::mem_row_major);
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
    int tiles_m = M / 16;
    int tiles_n = N / 16;
    int total_warps = tiles_m * tiles_n;
    
    // 每个 Block 包含 4 个 Warps (128 线程)
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * 32;
    int total_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    dim3 dimBlock(threads_per_block);
    dim3 dimGrid(total_blocks);
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