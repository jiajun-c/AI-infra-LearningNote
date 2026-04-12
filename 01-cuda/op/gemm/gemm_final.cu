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
// 设计：As[tidx][j] = A[row][i+j], Bs[j][tidy] = B[i+j][col]
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

// ===================== V3: 添加 padding 避免 bank conflict =====================
// Bs 有 padding，访问 Bs[j][tidy] 时不同 tidy 映射到不同 bank
__global__ void gemm_v3_padding(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int row = bidx * TILE_SIZE + tidx;
    int col = bidy * TILE_SIZE + tidy;
    float acc = 0.0f;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
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

// ===================== V4: 转置 B 存储 =====================
// Bs[tidy][tidx] 存储 B[i+tidx][col]，访问 Bs[tidy][j] 无 bank conflict
__global__ void gemm_v4_transB(float* A, float* B, float* C, int M, int N, int K) {
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
        Bs[tidy][tidx] = (i + tidx < K) ? B[(i + tidx) * N + col] : 0.0f;  // 转置存储
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) acc += As[tidx][j] * Bs[tidy][j];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ===================== V5: 每个线程计算 2x2 元素 =====================
__global__ void gemm_v5_2x2(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int base_row = bidx * TILE_SIZE + tidx * 2;
    int base_col = bidy * TILE_SIZE + tidy * 2;
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    __shared__ float As[TILE_SIZE * 2][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * 2 + 1];
    bool valid_row0 = (base_row < M);
    bool valid_row1 = (base_row + 1 < M);
    bool valid_col0 = (base_col < N);
    bool valid_col1 = (base_col + 1 < N);
    for (int i = 0; i < K; i += TILE_SIZE) {
        if (valid_row0)
            for (int k = 0; k < TILE_SIZE; k++)
                As[tidx * 2][k] = (i + k < K) ? A[base_row * K + i + k] : 0.0f;
        if (valid_row1)
            for (int k = 0; k < TILE_SIZE; k++)
                As[tidx * 2 + 1][k] = (i + k < K) ? A[(base_row + 1) * K + i + k] : 0.0f;
        if (valid_col0)
            for (int k = 0; k < TILE_SIZE; k++)
                Bs[k][tidx] = (i + k < K) ? B[(i + k) * N + base_col] : 0.0f;
        if (valid_col1)
            for (int k = 0; k < TILE_SIZE; k++)
                Bs[k][tidx + TILE_SIZE] = (i + k < K) ? B[(i + k) * N + base_col + 1] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) {
            acc00 += As[tidx * 2][j] * Bs[j][tidx];
            acc01 += As[tidx * 2][j] * Bs[j][tidx + TILE_SIZE];
            acc10 += As[tidx * 2 + 1][j] * Bs[j][tidx];
            acc11 += As[tidx * 2 + 1][j] * Bs[j][tidx + TILE_SIZE];
        }
        __syncthreads();
    }
    if (valid_row0 && valid_col0) C[base_row * N + base_col] = acc00;
    if (valid_row0 && valid_col1) C[base_row * N + base_col + 1] = acc01;
    if (valid_row1 && valid_col0) C[(base_row + 1) * N + base_col] = acc10;
    if (valid_row1 && valid_col1) C[(base_row + 1) * N + base_col + 1] = acc11;
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

// ===================== Verification =====================
bool verify_gemm(const float* gpu, const float* cpu, int M, int N, float atol = 1e-4f) {
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        max_err = fmaxf(max_err, err);
        if (err > atol) errors++;
    }
    printf("  Max error: %.2e, Mismatches: %d / %d\n", max_err, errors, M * N);
    return errors == 0;
}

// ===================== Benchmark =====================
typedef void (*gemm_fn)(float*, float*, float*, int, int, int);

float benchmark(gemm_fn kernel, float* d_A, float* d_B, float* d_C,
                int M, int N, int K, dim3 block, dim3 grid) {
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100;
}

// ===================== Main =====================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("========================================\n");
    printf("GPU: %s\n", prop.name);
    printf("========================================\n\n");

    struct TestConfig { int M, N, K; const char* desc; } configs[] = {
        {64, 64, 64, "64x64x64"},
        {128, 128, 128, "128x128x128"},
        {256, 256, 256, "256x256x256"},
        {512, 512, 512, "512x512x512"},
        {1024, 1024, 1024, "1024x1024x1024"},
    };

    struct KernelEntry {
        gemm_fn fn;
        const char* name;
        dim3 block;
    } kernels[] = {
        {gemm_v2_correct, "V2 (correct)"},
        {gemm_v3_padding, "V3 (padding)"},
        {gemm_v4_transB,  "V4 (transB)"},
        {gemm_v5_2x2,     "V5 (2x2)"},
    };
    kernels[0].block = dim3(16, 16);
    kernels[1].block = dim3(16, 16);
    kernels[2].block = dim3(16, 16);
    kernels[3].block = dim3(8, 8);

    // Debug: print kernel addresses
    printf("Kernel addresses: v2=%p, v3=%p, v4=%p, v5=%p\n",
           (void*)gemm_v2_correct, (void*)gemm_v3_padding,
           (void*)gemm_v4_transB, (void*)gemm_v5_2x2);
    for (int i = 0; i < 4; i++) {
        printf("  kernels[%d].fn=%p, block=(%d,%d)\n",
               i, (void*)kernels[i].fn, kernels[i].block.x, kernels[i].block.y);
    }

    for (auto& cfg : configs) {
        int M = cfg.M, N = cfg.N, K = cfg.K;
        printf("===== Test: %s =====\n", cfg.desc);

        float *h_A = (float*)malloc(M * K * sizeof(float));
        float *h_B = (float*)malloc(K * N * sizeof(float));
        float *h_C_gpu = (float*)malloc(M * N * sizeof(float));
        float *h_C_cpu = (float*)malloc(M * N * sizeof(float));

        srand(42);
        for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2 - 1;

        gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

        for (auto& k : kernels) {
            dim3 grid((M + k.block.x - 1) / k.block.x, (N + k.block.y - 1) / k.block.y);
            printf("\n  %s: grid=(%d,%d), block=(%d,%d), fn=%p\n",
                   k.name, grid.x, grid.y, k.block.x, k.block.y, (void*)k.fn);

            cudaMemset(d_C, 0, M * N * sizeof(float));
            k.fn(d_A, d_B, d_C, M, N, K);
            cudaDeviceSynchronize();
            cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            printf("  Correctness: %s\n", verify_gemm(h_C_gpu, h_C_cpu, M, N) ? "PASS" : "FAIL");

            float ms = benchmark(k.fn, d_A, d_B, d_C, M, N, K, k.block, grid);
            float gflops = (2.0f * M * N * K) / (ms * 1e-3f) / 1e9f;
            printf("  Time: %.4f ms, GFLOPS: %.2f\n", ms, gflops);
        }
        printf("\n");

        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    return 0;
}
