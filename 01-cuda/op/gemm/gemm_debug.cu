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

// ===================== V2: 正确版本 =====================
__global__ void gemm_v2_correct(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int row = bidx * TILE_SIZE + tidx;
    int col = bidy * TILE_SIZE + tidy;
    float acc = 0.0f;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    if (row >= M || col >= N) return;
    for (int i = 0; i < K; i += TILE_SIZE) {
        As[tidx][tidy] = (i + tidy < K) ? A[row * K + i + tidy] : 0.0f;
        Bs[tidx][tidy] = (i + tidx < K) ? B[(i + tidx) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) acc += As[tidx][j] * Bs[j][tidy];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ===================== CPU Reference =====================
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}

int main() {
    int M = 64, N = 64, K = 64;

    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_gpu = (float*)malloc(M * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);

    printf("CPU C[0][0:5] = ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_C_cpu[i]);
    printf("\n");

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    printf("grid=(%d,%d), block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);

    cudaMemset(d_C, 0, M * N * sizeof(float));
    gemm_v2_correct<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU C[0][0:5] = ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_C_gpu[i]);
    printf("\n");

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) {
            errors++;
            if (errors <= 5) {
                printf("Error at [%d][%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i/N, i%N, h_C_gpu[i], h_C_cpu[i], fabsf(h_C_gpu[i] - h_C_cpu[i]));
            }
        }
    }
    printf("Total errors: %d / %d\n", errors, M * N);

    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return errors > 0 ? 1 : 0;
}
