#include <iostream>
#include <vector>
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

int main() {
    const int N = 512;
    const int M = 1024;
    const int blocks = 132;
    const int threads = 256;

    size_t size_A = N * M * sizeof(float);
    size_t size_X = M * sizeof(float);
    size_t size_Y = N * sizeof(float);

    float *d_A, *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_X, size_X));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y));

    std::vector<float> h_A(N * M, 1.0f);
    std::vector<float> h_X(M, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 2; i++) {
        gemv_smid_based<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, blocks);
        gemv_warpid_based<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, blocks);
        gemv_standard<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
        gemv_simple<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
        cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 每个 kernel 跑 2 次
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Running kernels for profiling..." << std::endl;

    // SMID
    for (int i = 0; i < 2; i++) {
        cudaEventRecord(start);
        gemv_smid_based<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, blocks);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "SMID iteration " << i+1 << ": " << std::fixed << std::setprecision(6) << ms << " ms" << std::endl;
    }

    // WarpID
    for (int i = 0; i < 2; i++) {
        cudaEventRecord(start);
        gemv_warpid_based<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, blocks);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "WarpID iteration " << i+1 << ": " << std::fixed << std::setprecision(6) << ms << " ms" << std::endl;
    }

    // Standard
    for (int i = 0; i < 2; i++) {
        cudaEventRecord(start);
        gemv_standard<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Standard iteration " << i+1 << ": " << std::fixed << std::setprecision(6) << ms << " ms" << std::endl;
    }

    // Simple
    for (int i = 0; i < 2; i++) {
        cudaEventRecord(start);
        gemv_simple<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Simple iteration " << i+1 << ": " << std::fixed << std::setprecision(6) << ms << " ms" << std::endl;
    }

    // cuBLAS
    for (int i = 0; i < 2; i++) {
        cudaEventRecord(start);
        cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "cuBLAS iteration " << i+1 << ": " << std::fixed << std::setprecision(6) << ms << " ms" << std::endl;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(cublas_handle);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));

    std::cout << "Done. Profile output ready for ncu." << std::endl;
    return 0;
}
