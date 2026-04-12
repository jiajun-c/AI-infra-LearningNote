#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

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

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}

int main() {
    int M=64, N=64, K=64;
    float *h_A = new float[M*K], *h_B = new float[K*N], *h_C_gpu = new float[M*N], *h_C_cpu = new float[M*N];
    for (int i = 0; i < M*K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K*N; i++) h_B[i] = 1.0f;
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    printf("CPU C[0][0] = %.2f (expected %.2f)\n", h_C_cpu[0], (float)K);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16), grid(4,4);
    gemm_v2<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("GPU C[0][0] = %.2f\n", h_C_gpu[0]);
    int errors = 0;
    for (int i = 0; i < M*N; i++) if (h_C_gpu[i] != h_C_cpu[i]) errors++;
    printf("Errors: %d / %d\n", errors, M*N);
    return 0;
}
