#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
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

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------
// Kernel 1: SMID-based Fused GEMV + Activation
// GEMV 和 Activation 在同一个 SM 上完成，数据保持在 L2/cache 中
// ---------------------------------------------------------
__global__ void fused_gemv_relu_smid(const float* __restrict__ A,
                                     const float* __restrict__ X,
                                     float* __restrict__ Y,
                                     int N, int M,
                                     int sm_count,
                                     bool apply_activation) {
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
        if (laneID == 0) {
            if (apply_activation) {
                Y[row] = relu(partial_sum);
            } else {
                Y[row] = partial_sum;
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 2: WarpID-based Fused GEMV + Activation (对照组)
// ---------------------------------------------------------
__global__ void fused_gemv_relu_warpid(const float* __restrict__ A,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       int N, int M,
                                       int active_blocks,
                                       bool apply_activation) {
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
        if (laneID == 0) {
            if (apply_activation) {
                Y[row] = relu(partial_sum);
            } else {
                Y[row] = partial_sum;
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 3: Standard GEMV (对照组，不含 activation)
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// Kernel 4: Activation Kernel (单独的 ReLU)
// ---------------------------------------------------------
__global__ void apply_relu(float* Y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = relu(Y[idx]);
    }
}

// ---------------------------------------------------------
// CPU 参考实现
// ---------------------------------------------------------
void gemv_cpu_reference(const float* A, const float* X, float* Y, int N, int M) {
    for (int i = 0; i < N; i++) {
        Y[i] = 0.0f;
        for (int j = 0; j < M; j++) {
            Y[i] += A[i * M + j] * X[j];
        }
    }
}

void apply_relu_cpu(float* Y, int N) {
    for (int i = 0; i < N; i++) {
        Y[i] = fmaxf(0.0f, Y[i]);
    }
}

enum KernelType { SMID_FUSED, WARPID_FUSED, STANDARD_SEPARATE, CUBLAS_SEPARATE };

float test_fused_gemv(KernelType type, const float* d_A, const float* d_X, float* d_Y,
                      int N, int M, int blocks_limit, int threads_per_block,
                      cublasHandle_t cublas_handle,
                      float* d_Y_tmp) {
    int launch_blocks = 132;
    bool apply_activation = true;

    if (type == CUBLAS_SEPARATE) {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N,
                                  &alpha, d_A, M, d_X, 1, &beta, d_Y, 1));

        // 单独的 ReLU kernel
        int relu_blocks = (N + 256 - 1) / 256;
        apply_relu<<<relu_blocks, 256>>>(d_Y, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N,
                        &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
            apply_relu<<<relu_blocks, 256>>>(d_Y, N);
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / iterations;

    } else if (type == STANDARD_SEPARATE) {
        // GEMV + 单独的 ReLU
        gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        int relu_blocks = (N + 256 - 1) / 256;
        apply_relu<<<relu_blocks, 256>>>(d_Y, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
            apply_relu<<<relu_blocks, 256>>>(d_Y, N);
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / iterations;

    } else {
        if (type == SMID_FUSED) {
            fused_gemv_relu_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit, apply_activation);
        } else {
            fused_gemv_relu_warpid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit, apply_activation);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            if (type == SMID_FUSED) {
                fused_gemv_relu_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit, apply_activation);
            } else {
                fused_gemv_relu_warpid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit, apply_activation);
            }
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / iterations;
    }
}

bool verify_fused_gemv(const float* d_A, const float* d_X, float* d_Y,
                       int N, int M, cublasHandle_t cublas_handle) {
    std::vector<float> h_Y_cpu(N, 0.0f);
    std::vector<float> h_A(N * M);
    std::vector<float> h_X(M);

    CHECK_CUDA(cudaMemcpy(h_A.data(), d_A, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_X.data(), d_X, M * sizeof(float), cudaMemcpyDeviceToHost));

    gemv_cpu_reference(h_A.data(), h_X.data(), h_Y_cpu.data(), N, M);
    apply_relu_cpu(h_Y_cpu.data(), N);

    std::vector<float> h_Y_gpu(N);
    CHECK_CUDA(cudaMemcpy(h_Y_gpu.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float diff = std::abs(h_Y_cpu[i] - h_Y_gpu[i]);
        float rel_error = diff / (std::abs(h_Y_cpu[i]) + 1e-8f);
        if (rel_error > 1e-4f && diff > 1e-4f) {
            std::cerr << "验证失败：位置 " << i << ", 期望=" << h_Y_cpu[i]
                      << ", 实际=" << h_Y_gpu[i] << ", 相对误差=" << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<std::pair<int, int>> test_sizes = {
        {1024, 128},
        {1024, 256},
        {1024, 512},
        {2048, 128},
        {2048, 256},
        {2048, 512},
        {4096, 128},
        {4096, 256},
        {8192, 128}
    };

    int threads_per_block = 256;

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    std::cout << "=== Fused GEMV + ReLU 性能测试 ===" << std::endl;
    std::cout << "比较 Fused (GEMV+ReLU 单 kernel) vs Separate (GEMV + ReLU 双 kernel)" << std::endl;
    std::cout << std::endl;

    for (auto& [N, M] : test_sizes) {
        size_t size_A = N * M * sizeof(float);
        size_t size_X = M * sizeof(float);
        size_t size_Y = N * sizeof(float);

        std::vector<float> h_A(N * M, 1.0f);
        std::vector<float> h_X(M, 1.0f);

        float *d_A, *d_X, *d_Y, *d_Y_tmp;
        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMalloc(&d_X, size_X));
        CHECK_CUDA(cudaMalloc(&d_Y, size_Y));
        CHECK_CUDA(cudaMalloc(&d_Y_tmp, size_Y));
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice));

        // 正确性验证
        fused_gemv_relu_smid<<<132, threads_per_block>>>(d_A, d_X, d_Y, N, M, 132, true);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass = verify_fused_gemv(d_A, d_X, d_Y, N, M, cublas_handle);

        double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;

        float ms_smid = test_fused_gemv(SMID_FUSED, d_A, d_X, d_Y, N, M, 132, threads_per_block, cublas_handle, d_Y_tmp);
        float ms_warpid = test_fused_gemv(WARPID_FUSED, d_A, d_X, d_Y, N, M, 132, threads_per_block, cublas_handle, d_Y_tmp);
        float ms_standard = test_fused_gemv(STANDARD_SEPARATE, d_A, d_X, d_Y, N, M, 132, threads_per_block, cublas_handle, d_Y_tmp);
        float ms_cublas = test_fused_gemv(CUBLAS_SEPARATE, d_A, d_X, d_Y, N, M, 132, threads_per_block, cublas_handle, d_Y_tmp);

        double bw_smid = (bytes / (ms_smid / 1000.0)) / 1e9;
        double bw_warpid = (bytes / (ms_warpid / 1000.0)) / 1e9;
        double bw_standard = (bytes / (ms_standard / 1000.0)) / 1e9;
        double bw_cublas = (bytes / (ms_cublas / 1000.0)) / 1e9;

        float fused_speedup = (bw_smid - bw_standard) / bw_standard * 100;

        std::cout << std::left << std::setw(12) << (std::to_string(N) + "x" + std::to_string(M))
                  << "  验证=" << (pass ? "PASS" : "FAIL")
                  << "  " << std::setw(14) << (std::to_string((int)(bw_smid * 100) / 100.0) + " GB/s")
                  << "  " << std::setw(14) << (std::to_string((int)(bw_warpid * 100) / 100.0) + " GB/s")
                  << "  " << std::setw(14) << (std::to_string((int)(bw_standard * 100) / 100.0) + " GB/s")
                  << "  " << std::setw(14) << (std::to_string((int)(bw_cublas * 100) / 100.0) + " GB/s")
                  << "  Fused 提升=" << std::setw(8) << (std::to_string((int)(fused_speedup * 100) / 100.0) + "%")
                  << std::endl;

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Y));
        CHECK_CUDA(cudaFree(d_Y_tmp));
    }

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    return 0;
}
