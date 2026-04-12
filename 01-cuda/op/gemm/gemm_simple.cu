#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cfloat>

using namespace std;

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ===================== 标准 tiled GEMM =====================
// 每个 block 计算 C 的一个 TILE x TILE 块
// 每个线程计算 C 的一个元素
// C[row][col] = sum_k(A[row][k] * B[k][col])
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    // 计算此线程负责计算的 C 的全局坐标
    int row = bidx * TILE_SIZE + tidx;
    int col = bidy * TILE_SIZE + tidy;

    // 累加器（寄存器）
    float acc = 0.0f;

    // 共享内存：存储 A 和 B 的 TILE_SIZE x TILE_SIZE 块
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 边界检查
    if (row >= M || col >= N) return;

    // K 维度分块迭代
    for (int i = 0; i < K; i += TILE_SIZE) {
        // 加载 A 块：As[tidx][tidy] = A[row][i + tidy]
        // 注意：这里用 tidy 作为 A 的列索引偏移
        if (i + tidy < K) {
            As[tidx][tidy] = A[row * K + i + tidy];
        } else {
            As[tidx][tidy] = 0.0f;
        }

        // 加载 B 块：Bs[tidx][tidy] = B[i + tidx][col]
        // 注意：这里用 tidx 作为 B 的行索引偏移
        if (i + tidx < K) {
            Bs[tidx][tidy] = B[(i + tidx) * N + col];
        } else {
            Bs[tidx][tidy] = 0.0f;
        }

        __syncthreads();

        // 计算：acc += A[row][i+j] * B[i+j][col]
        // As[tidx][j] = A[row][i+j]
        // Bs[j][tidy] = B[i+j][col]  <-- 注意这里！
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) {
            acc += As[tidx][j] * Bs[j][tidy];
        }

        __syncthreads();
    }

    // 写回结果
    C[row * N + col] = acc;
}

// ===================== CPU Reference =====================
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// ===================== Main =====================
int main() {
    int M = 64, N = 64, K = 64;
    int total_A = M * K, total_B = K * N, total_C = M * N;

    // 分配 Host 内存
    float *h_A = (float*)malloc(total_A * sizeof(float));
    float *h_B = (float*)malloc(total_B * sizeof(float));
    float *h_C_gpu = (float*)malloc(total_C * sizeof(float));
    float *h_C_cpu = (float*)malloc(total_C * sizeof(float));

    // 初始化
    srand(42);
    for (int i = 0; i < total_A; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < total_B; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // CPU 计算
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);

    // 打印一些 CPU 结果用于调试
    printf("CPU C[0][0:5] = ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_C_cpu[i]);
    printf("\n");

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, total_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, total_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, total_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, total_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, total_B * sizeof(float), cudaMemcpyHostToDevice));

    // 启动 kernel
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    printf("Launching kernel with grid=(%d,%d), block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);

    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回 Host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, total_C * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印 GPU 结果
    printf("GPU C[0][0:5] = ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_C_gpu[i]);
    printf("\n");

    // 验证
    int errors = 0;
    for (int i = 0; i < total_C; i++) {
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-3f) {
            errors++;
            if (errors <= 5) {
                printf("Error at [%d][%d]: GPU=%.6f, CPU=%.6f\n", i/N, i%N, h_C_gpu[i], h_C_cpu[i]);
            }
        }
    }
    printf("Total errors: %d / %d\n", errors, total_C);

    // 清理
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return errors > 0 ? 1 : 0;
}
