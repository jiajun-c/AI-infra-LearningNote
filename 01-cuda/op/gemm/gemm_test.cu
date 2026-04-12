#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// V2: Correct version
__global__ void gemm_v2(float* A, float* B, float* C, int M, int N, int K) {
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
        for (int j = 0; j < TILE_SIZE; j++) acc += As[tidx][j] * Bs[j][tidy];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// V3: With padding
__global__ void gemm_v3(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int row = bidx * TILE_SIZE + tidx;
    int col = bidy * TILE_SIZE + tidy;
    float acc = 0.0f;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    if (row >= M || col >= N) return;
    for (int i = 0; i < K; i += TILE_SIZE) {
        As[tidx][tidy] = (i + tidy < K) ? A[row * K + i + tidy] : 0.0f;
        Bs[tidx][tidy] = (i + tidx < K) ? B[(i + tidx) * N + col] : 0.0f;
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++) acc += As[tidx][j] * Bs[j][tidy];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// V4: Transposed B
__global__ void gemm_v4(float* A, float* B, float* C, int M, int N, int K) {
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
        Bs[tidy][tidx] = (i + tidx < K) ? B[(i + tidx) * N + col] : 0.0f;
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++) acc += As[tidx][j] * Bs[tidy][j];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}

bool verify(const float* gpu, const float* cpu, int total) {
    int errors = 0;
    for (int i = 0; i < total; i++)
        if (fabsf(gpu[i] - cpu[i]) > 1e-4f) errors++;
    return errors == 0;
}

int main() {
    int M=64, N=64, K=64;
    float *h_A = new float[M*K], *h_B = new float[K*N], *h_C_gpu = new float[M*N], *h_C_cpu = new float[M*N];
    srand(42);
    for (int i = 0; i < M*K; i++) h_A[i] = ((float)rand()/RAND_MAX)*2-1;
    for (int i = 0; i < K*N; i++) h_B[i] = ((float)rand()/RAND_MAX)*2-1;
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16), grid(4,4);

    cudaMemset(d_C, 0, M*N*sizeof(float));
    gemm_v2<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("V2: %s\n", verify(h_C_gpu, h_C_cpu, M*N) ? "PASS" : "FAIL");

    cudaMemset(d_C, 0, M*N*sizeof(float));
    gemm_v3<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("V3: %s\n", verify(h_C_gpu, h_C_cpu, M*N) ? "PASS" : "FAIL");

    cudaMemset(d_C, 0, M*N*sizeof(float));
    gemm_v4<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("V4: %s\n", verify(h_C_gpu, h_C_cpu, M*N) ? "PASS" : "FAIL");

    delete[] h_A; delete[] h_B; delete[] h_C_gpu; delete[] h_C_cpu;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
