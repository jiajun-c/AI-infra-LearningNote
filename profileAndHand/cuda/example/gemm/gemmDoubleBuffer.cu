#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
#define BLOCK_SIZE 32
__global__ void matmul(
        int M,
    int N,
    int K,
    const float* a,
    const float* b,
    float* c
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tileA[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[2][BLOCK_SIZE][BLOCK_SIZE];
    float res = 0;

    int Acol = threadIdx.x;
    int Brow = threadIdx.y;
    if (row < M && Acol < K) {
        tileA[0][threadIdx.y][threadIdx.x] = a[row*K + Acol];
    } else {
        tileA[0][threadIdx.y][threadIdx.x] = 0.0;
    }

    if (Brow < K && col < N) {
        tileB[0][threadIdx.y][threadIdx.x] = b[Brow*N + col];
    } else {
        tileB[0][threadIdx.y][threadIdx.x] = 0.0;
    }
    int iter = (K + BLOCK_SIZE - 1)/BLOCK_SIZE;
    for (int i = 0; i < iter; i++) {
        __syncthreads();
        int cur = i % 2;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            res += tileA[cur][threadIdx.y][j] * tileB[cur][j][threadIdx.x];
        }
        int next = (i + 1)%2;
        if (i + 1 < iter) {
            Acol = (i+1)*BLOCK_SIZE + threadIdx.x;
            Brow = (i+1)*BLOCK_SIZE + threadIdx.y;
            if (row < M && Acol < K) {
                tileA[next][threadIdx.y][threadIdx.x] = a[row*K + Acol];
            } else {
                tileA[next][threadIdx.y][threadIdx.x] = 0.0;
            }

            if (Brow < K && col < N) {
                tileB[next][threadIdx.y][threadIdx.x] = b[Brow*N + col];
            } else {
                tileB[next][threadIdx.y][threadIdx.x] = 0.0;
            }
        }
    }
    if (row < M  && col < N) {
        c[row * N + col] = res;
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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
    printf("Performance: %.2f TFLOPS\n", (2.0 * M * N * K * sizeof(float) / 1e12) / (milliseconds / 1000.0));
    printf("GPU Time: %.4f ms\n", milliseconds);

    // 拷贝回结果 D2H
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 校验结果
    printf("Verifying result...\n");
    cpu_gemm(M, N, K, h_A, h_B, h_C_ref);

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max Diff: %f\n", max_diff);
    if (max_diff < 1e-3) printf("✅ Test Passed!\n");
    else printf("❌ Test Failed!\n");

    // 清理
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}