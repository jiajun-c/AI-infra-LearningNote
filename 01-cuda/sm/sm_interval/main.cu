#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CPU 参考实现
void gemv_cpu_reference(const float* A, const float* X, float* Y, int N, int M) {
    for (int i = 0; i < N; i++) {
        Y[i] = 0.0f;
        for (int j = 0; j < M; j++) {
            Y[i] += A[i * M + j] * X[j];
        }
    }
}

__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// Kernel 1: Interleaved SM (交错 SM 放置)
__global__ void gemv_interval_sm(const float* __restrict__ A,
                                 const float* __restrict__ X,
                                 float* __restrict__ Y,
                                 int N, int M, int sm_count) {
    int smid = get_smid();

    if (smid % 2 != 0) return;  // 过滤奇数 SM

    int target_sm_idx = smid / 2;
    if (target_sm_idx >= sm_count) return;

    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    int global_warpID = target_sm_idx * warps_per_block + warpID;
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
            Y[row] = partial_sum;
        }
    }
}

// Kernel 2: Sequential SM (连续 SM 放置)
__global__ void gemv_seq_sm(const float* __restrict__ A,
                            const float* __restrict__ X,
                            float* __restrict__ Y,
                            int N, int M, int sm_count) {
    int smid = get_smid();
    if (smid >= sm_count) return;

    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;

    int target_id = smid * 8 + warpID;
    int full_warp = sm_count * 8;

    for (int row = target_id; row < N; row += full_warp) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) {
            Y[row] = partial_sum;
        }
    }
}

// Kernel 3: Standard (基于 blockIdx 的任务分配，但限制只使用 66 个 blocks)
__global__ void gemv_seq_standard(const float* __restrict__ A,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y,
                                  int N, int M, int target_blocks) {
    // 关键：限制只有前 target_blocks 个 blocks 工作
    if (blockIdx.x >= target_blocks) return;

    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    int global_warp_id = blockIdx.x * warps_per_block + warpID;
    int total_warps = target_blocks * warps_per_block;

    for (int row = global_warp_id; row < N; row += total_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) {
            Y[row] = partial_sum;
        }
    }
}

// ---------------------------------------------------------
// Host 端测试代码
// ---------------------------------------------------------

bool verify_kernel_sm(const char* name, void (*kernel)(const float*, const float*, float*, int, int, int),
                      const float* d_A, const float* d_X, float* d_Y,
                      int N, int M, int sm_count, int blocks, int threads,
                      const float* h_A, const float* h_X) {
    std::vector<float> h_Y_cpu(N, 0.0f);
    gemv_cpu_reference(h_A, h_X, h_Y_cpu.data(), N, M);

    std::vector<float> h_Y_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, sm_count);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_Y_gpu.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_Y_gpu[i] - h_Y_cpu[i]) > 1e-3f) {
            error_count++;
        }
    }

    if (error_count == 0) {
        std::cout << "  " << name << ": PASS" << std::endl;
        return true;
    } else {
        std::cout << "  " << name << ": FAIL (" << error_count << " errors)" << std::endl;
        return false;
    }
}

bool verify_kernel_std(const char* name, void (*kernel)(const float*, const float*, float*, int, int, int),
                       const float* d_A, const float* d_X, float* d_Y,
                       int N, int M, int blocks, int threads, int target_blocks,
                       const float* h_A, const float* h_X) {
    std::vector<float> h_Y_cpu(N, 0.0f);
    gemv_cpu_reference(h_A, h_X, h_Y_cpu.data(), N, M);

    std::vector<float> h_Y_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, target_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_Y_gpu.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_Y_gpu[i] - h_Y_cpu[i]) > 1e-3f) {
            error_count++;
        }
    }

    if (error_count == 0) {
        std::cout << "  " << name << ": PASS" << std::endl;
        return true;
    } else {
        std::cout << "  " << name << ": FAIL (" << error_count << " errors)" << std::endl;
        return false;
    }
}

float benchmark_kernel_sm(void (*kernel)(const float*, const float*, float*, int, int, int),
                          const float* d_A, const float* d_X, float* d_Y,
                          int N, int M, int sm_count, int blocks, int threads,
                          int warmup_iters, int test_iters) {
    for (int i = 0; i < warmup_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, sm_count);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < test_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, sm_count);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / test_iters;
}

float benchmark_kernel_std(void (*kernel)(const float*, const float*, float*, int, int, int),
                           const float* d_A, const float* d_X, float* d_Y,
                           int N, int M, int blocks, int threads, int target_blocks,
                           int warmup_iters, int test_iters) {
    for (int i = 0; i < warmup_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, target_blocks);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < test_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, target_blocks);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / test_iters;
}

int main() {
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    int total_sms = prop.multiProcessorCount;

    std::cout << "Device: " << prop.name << " | Total SMs: " << total_sms << std::endl;
    std::cout << "========================================================" << std::endl;

    std::vector<std::pair<int, int>> test_shapes = {
        // 小矩阵区域
        {256, 256},        // 0.25 MB
        {256, 512},        // 0.5 MB
        {512, 256},        // 0.5 MB
        {512, 512},        // 1 MB
        {512, 1024},       // 2 MB
        {1024, 512},       // 2 MB
        {1024, 1024},      // 4 MB
        {1024, 2048},      // 8 MB
        {2048, 1024},      // 8 MB
        {2048, 2048},      // 16 MB
        // 甜蜜点区域 - 细粒度测试
        {2048, 3072},      // 24 MB
        {2048, 4096},      // 32 MB (甜蜜点)
        {3072, 4096},      // 48 MB
        {4096, 2048},      // 32 MB
        {4096, 3072},      // 48 MB
        {4096, 4096},      // 64 MB
        // 大矩阵区域
        {4096, 8192},      // 128 MB
        {8192, 8192},      // 256 MB
        {8192, 16384},     // 512 MB
        {16384, 16384},    // 1024 MB
        {16384, 32768},    // 2048 MB
        {32768, 32768}     // 4096 MB
    };

    int target_sm_count = 66;
    int blocks = 66;  // Standard 版本需要 66 个 blocks
    int threads_per_block = 256;
    int warmup_iters = 10;
    int test_iters = 20;

    float *d_A, *d_X, *d_Y;

    std::cout << "Active SMs: " << target_sm_count << " (of " << total_sms << ")" << std::endl;
    std::cout << "Threads/block: " << threads_per_block << std::endl;
    std::cout << "========================================================" << std::endl;

    for (auto& shape : test_shapes) {
        int N = shape.first;
        int M = shape.second;

        size_t size_A = N * M * sizeof(float);
        size_t size_X = M * sizeof(float);
        size_t size_Y = N * sizeof(float);

        double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;
        double total_bytes_gb = bytes / (1024.0 * 1024.0 * 1024.0);

        std::vector<float> h_A(N * M, 1.0f);
        std::vector<float> h_X(M, 1.0f);

        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMalloc(&d_X, size_X));
        CHECK_CUDA(cudaMalloc(&d_Y, size_Y));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice));

        std::cout << "\n--------------------------------------------------------" << std::endl;
        std::cout << "Shape: " << N << " x " << M << " | Bytes: " << std::fixed << std::setprecision(4) << total_bytes_gb << " GB" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;

        // 正确性验证
        std::cout << "正确性验证：" << std::endl;
        bool seq_sm_pass = verify_kernel_sm("Sequential SM", gemv_seq_sm, d_A, d_X, d_Y, N, M, target_sm_count, 132, threads_per_block, h_A.data(), h_X.data());
        bool interval_pass = verify_kernel_sm("Interleaved SM", gemv_interval_sm, d_A, d_X, d_Y, N, M, target_sm_count, 132, threads_per_block, h_A.data(), h_X.data());

        if (!seq_sm_pass || !interval_pass) {
            std::cerr << "验证失败，跳过性能测试" << std::endl;
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_X));
            CHECK_CUDA(cudaFree(d_Y));
            continue;
        }

        // 性能测试
        float seq_sm_ms = benchmark_kernel_sm(gemv_seq_sm, d_A, d_X, d_Y, N, M, target_sm_count, 132, threads_per_block, warmup_iters, test_iters);
        double seq_sm_bw = (bytes / (seq_sm_ms / 1000.0)) / 1e9;

        float interval_ms = benchmark_kernel_sm(gemv_interval_sm, d_A, d_X, d_Y, N, M, target_sm_count, 132, threads_per_block, warmup_iters, test_iters);
        double interval_bw = (bytes / (interval_ms / 1000.0)) / 1e9;

        std::cout << "\n性能对比：" << std::endl;
        std::cout << "  Sequential SM (SM 0-65):    " << std::fixed << std::setprecision(4) << seq_sm_ms << " ms, "
                  << std::fixed << std::setprecision(2) << seq_sm_bw << " GB/s" << std::endl;
        std::cout << "  Interleaved SM (SM 0,2,..): " << std::fixed << std::setprecision(4) << interval_ms << " ms, "
                  << std::fixed << std::setprecision(2) << interval_bw << " GB/s" << std::endl;

        // 计算差异
        double int_vs_seq_diff = interval_bw - seq_sm_bw;
        double int_vs_seq_pct = (int_vs_seq_diff / seq_sm_bw) * 100.0;
        std::cout << "\n  Interleaved vs Sequential: " << (int_vs_seq_diff > 0 ? "+" : "") << std::fixed << std::setprecision(2)
                  << int_vs_seq_diff << " GB/s (" << (int_vs_seq_diff > 0 ? "+" : "") << std::setprecision(2) << int_vs_seq_pct << "%)" << std::endl;

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Y));
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "测试完成!" << std::endl;
    std::cout << "========================================================" << std::endl;

    return 0;
}
