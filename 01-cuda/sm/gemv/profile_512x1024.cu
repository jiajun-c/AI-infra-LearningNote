#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// Kernel 1: 基于 smid 控制
__global__ void gemv_smid_based(const float* __restrict__ A,
                                const float* __restrict__ X,
                                float* __restrict__ Y,
                                int N, int M, int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();
    if (my_smid >= sm_count) return;
    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = sm_count * warps_per_block;
    for (int row = global_warpID; row < N; row += total_active_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) Y[row] = partial_sum;
    }
}

// Kernel 2: 基于 warpID 控制
__global__ void gemv_warpid_based(const float* __restrict__ A,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y,
                                  int N, int M, int active_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = tid / 32;
    int laneID = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int max_active_warps = active_blocks * warps_per_block;
    if (warpID >= max_active_warps) return;
    for (int row = warpID; row < N; row += max_active_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) Y[row] = partial_sum;
    }
}

// Kernel 3: 标准 GEMV
__global__ void gemv_standard(const float* __restrict__ A,
                              const float* __restrict__ X,
                              float* __restrict__ Y,
                              int N, int M) {
    int rows_per_block = (N + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, N);
    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    for (int row = row_start + warpID; row < row_end; row += warps_per_block) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) Y[row] = partial_sum;
    }
}

// Kernel 4: 简单 GEMV
__global__ void gemv_simple(const float* __restrict__ A,
                            const float* __restrict__ X,
                            float* __restrict__ Y,
                            int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float sum = 0.0f;
    for (int col = threadIdx.x % 32; col < M; col += 32) {
        sum += A[row * M + col] * X[col];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x % 32 == 0) Y[row] = sum;
}

enum KernelType { SMID_BASED, WARPID_BASED, STANDARD, SIMPLE, CUBLAS };

float test_gemv_performance(KernelType type, const float* d_A, const float* d_X, float* d_Y,
                            int N, int M, int blocks_limit, int threads_per_block,
                            cublasHandle_t cublas_handle) {
    int launch_blocks = 132;

    if (type == CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T,
                                  M, N,
                                  &alpha,
                                  d_A, M,
                                  d_X, 1,
                                  &beta,
                                  d_Y, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        if (type == SMID_BASED) {
            gemv_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == WARPID_BASED) {
            gemv_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == STANDARD) {
            gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        } else {
            gemv_simple<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float alpha = 1.0f, beta = 0.0f;

    // Warmup 2 次
    for (int i = 0; i < 2; i++) {
        if (type == CUBLAS) {
            cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
        } else if (type == SMID_BASED) {
            gemv_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == WARPID_BASED) {
            gemv_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == STANDARD) {
            gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        } else {
            gemv_simple<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 正式跑 2 次
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    int iterations = 2;
    for (int iter = 0; iter < iterations; iter++) {
        cudaEventRecord(start);
        if (type == CUBLAS) {
            cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
        } else if (type == SMID_BASED) {
            gemv_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == WARPID_BASED) {
            gemv_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == STANDARD) {
            gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        } else {
            gemv_simple<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "  Iteration " << iter + 1 << ": " << std::fixed << std::setprecision(6) << ms << " ms" << std::endl;
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / iterations;
}

int main() {
    // 固定测试矩阵大小：512 x 1024
    const int N = 512;
    const int M = 1024;
    const int blocks_limit = 132;
    const int threads_per_block = 256;

    size_t size_A = N * M * sizeof(float);
    size_t size_X = M * sizeof(float);
    size_t size_Y = N * sizeof(float);

    double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;
    double total_bytes = bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << "========================================================" << std::endl;
    std::cout << "测试矩阵大小：" << N << " x " << M << " (FP32)" << std::endl;
    std::cout << "访存量：~" << std::fixed << std::setprecision(6) << total_bytes << " GB" << std::endl;
    std::cout << "========================================================" << std::endl;

    // 初始化 cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // 分配显存
    float *d_A, *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_X, size_X));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y));

    // 初始化数据
    std::vector<float> h_A(N * M, 1.0f);
    std::vector<float> h_X(M, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice));

    // 测试各个版本
    KernelType kernels[] = {SMID_BASED, WARPID_BASED, STANDARD, SIMPLE, CUBLAS};
    const char* kernel_names[] = {"SMID", "WarpID", "Standard", "Simple", "cuBLAS"};

    std::cout << "\n各版本性能 (warmup=2, runs=2):" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int i = 0; i < 5; i++) {
        std::cout << "\n" << kernel_names[i] << ":" << std::endl;
        float avg_ms = test_gemv_performance(kernels[i], d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle);
        double bandwidth = (bytes / (avg_ms / 1000.0)) / 1e9;
        std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms << " ms" << std::endl;
        std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    }

    std::cout << "\n========================================================" << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    cublasDestroy(cublas_handle);

    return 0;
}
