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

// Kernel 1: 基于 SMID 的任务分配 (使用全部 132 个 SM)
__global__ void gemv_smid_based(const float* __restrict__ A,
                                const float* __restrict__ X,
                                float* __restrict__ Y,
                                int N, int M, int total_sms) {
    unsigned int smid = get_smid();
    if (smid >= total_sms) return;

    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    // 每个 SM 有多个 warps，计算全局 warp ID
    int global_warpID = smid * warps_per_block + warpID;
    int total_active_warps = total_sms * warps_per_block;

    // Grid-stride loop: 每个 warp 负责多行
    for (int row = global_warpID; row < N; row += total_active_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        // Warp 内归约
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) {
            Y[row] = partial_sum;
        }
    }
}

// Kernel 2: 基于 blockIdx.x 的任务分配 (标准写法)
__global__ void gemv_blockidx_based(const float* __restrict__ A,
                                    const float* __restrict__ X,
                                    float* __restrict__ Y,
                                    int N, int M, int total_sms) {
    // 假设 1 block = 1 SM
    int smid = blockIdx.x;
    if (smid >= total_sms) return;

    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    // 每个 SM 有多个 warps，计算全局 warp ID
    int global_warpID = smid * warps_per_block + warpID;
    int total_active_warps = total_sms * warps_per_block;

    // Grid-stride loop: 每个 warp 负责多行
    for (int row = global_warpID; row < N; row += total_active_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        // Warp 内归约
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) {
            Y[row] = partial_sum;
        }
    }
}

// Kernel 3: 基于 blockIdx.x 的另一种写法 (row 在 block 内分配)
__global__ void gemv_blockidx_row_based(const float* __restrict__ A,
                                        const float* __restrict__ X,
                                        float* __restrict__ Y,
                                        int N, int M) {
    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    // 每个 block 负责一部分行
    int rows_per_block = (N + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, N);

    // Block 内的 warp 分配行
    for (int row = row_start + warpID; row < row_end; row += warps_per_block) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        // Warp 内归约
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

bool verify_kernel(const char* name, void (*kernel)(const float*, const float*, float*, int, int, int),
                   const float* d_A, const float* d_X, float* d_Y,
                   int N, int M, int total_sms, int blocks, int threads,
                   const float* h_A, const float* h_X) {
    std::vector<float> h_Y_cpu(N, 0.0f);
    gemv_cpu_reference(h_A, h_X, h_Y_cpu.data(), N, M);

    std::vector<float> h_Y_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, total_sms);
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

bool verify_kernel_no_sm_param(const char* name, void (*kernel)(const float*, const float*, float*, int, int),
                                const float* d_A, const float* d_X, float* d_Y,
                                int N, int M, int blocks, int threads,
                                const float* h_A, const float* h_X) {
    std::vector<float> h_Y_cpu(N, 0.0f);
    gemv_cpu_reference(h_A, h_X, h_Y_cpu.data(), N, M);

    std::vector<float> h_Y_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
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

float benchmark_kernel(const float* d_A, const float* d_X, float* d_Y,
                       void (*kernel)(const float*, const float*, float*, int, int, int),
                       int N, int M, int total_sms, int blocks, int threads,
                       int warmup_iters, int test_iters) {
    for (int i = 0; i < warmup_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, total_sms);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < test_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M, total_sms);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / test_iters;
}

float benchmark_kernel_no_sm_param(const float* d_A, const float* d_X, float* d_Y,
                                    void (*kernel)(const float*, const float*, float*, int, int),
                                    int N, int M, int blocks, int threads,
                                    int warmup_iters, int test_iters) {
    for (int i = 0; i < warmup_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < test_iters; i++) {
        kernel<<<blocks, threads>>>(d_A, d_X, d_Y, N, M);
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
        // 甜蜜点区域
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

    int blocks = total_sms;  // 使用全部 SM
    int threads_per_block = 256;
    int warmup_iters = 10;
    int test_iters = 20;

    float *d_A, *d_X, *d_Y;

    std::cout << "Active SMs: " << blocks << " (of " << total_sms << ")" << std::endl;
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
        bool smid_pass = verify_kernel("SMID-based", gemv_smid_based, d_A, d_X, d_Y, N, M, total_sms, blocks, threads_per_block, h_A.data(), h_X.data());
        bool blockidx_pass = verify_kernel("BlockIdx-based", gemv_blockidx_based, d_A, d_X, d_Y, N, M, total_sms, blocks, threads_per_block, h_A.data(), h_X.data());
        bool blockidx_row_pass = verify_kernel_no_sm_param("BlockIdx-row-based", gemv_blockidx_row_based, d_A, d_X, d_Y, N, M, blocks, threads_per_block, h_A.data(), h_X.data());

        if (!smid_pass || !blockidx_pass || !blockidx_row_pass) {
            std::cerr << "验证失败，跳过性能测试" << std::endl;
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_X));
            CHECK_CUDA(cudaFree(d_Y));
            continue;
        }

        // 性能测试
        float smid_ms = benchmark_kernel(d_A, d_X, d_Y, gemv_smid_based, N, M, total_sms, blocks, threads_per_block, warmup_iters, test_iters);
        double smid_bw = (bytes / (smid_ms / 1000.0)) / 1e9;

        float blockidx_ms = benchmark_kernel(d_A, d_X, d_Y, gemv_blockidx_based, N, M, total_sms, blocks, threads_per_block, warmup_iters, test_iters);
        double blockidx_bw = (bytes / (blockidx_ms / 1000.0)) / 1e9;

        float blockidx_row_ms = benchmark_kernel_no_sm_param(d_A, d_X, d_Y, gemv_blockidx_row_based, N, M, blocks, threads_per_block, warmup_iters, test_iters);
        double blockidx_row_bw = (bytes / (blockidx_row_ms / 1000.0)) / 1e9;

        std::cout << "\n性能对比：" << std::endl;
        std::cout << "  SMID-based (smid 寄存器):     " << std::fixed << std::setprecision(4) << smid_ms << " ms, "
                  << std::fixed << std::setprecision(2) << smid_bw << " GB/s" << std::endl;
        std::cout << "  BlockIdx-based (blockIdx.x):  " << std::fixed << std::setprecision(4) << blockidx_ms << " ms, "
                  << std::fixed << std::setprecision(2) << blockidx_bw << " GB/s" << std::endl;
        std::cout << "  BlockIdx-row-based (行分配):  " << std::fixed << std::setprecision(4) << blockidx_row_ms << " ms, "
                  << std::fixed << std::setprecision(2) << blockidx_row_bw << " GB/s" << std::endl;

        // 计算差异
        double blockidx_vs_smid_diff = blockidx_bw - smid_bw;
        double blockidx_vs_smid_pct = (blockidx_vs_smid_diff / smid_bw) * 100.0;
        std::cout << "\n  BlockIdx vs SMID: " << (blockidx_vs_smid_diff > 0 ? "+" : "") << std::fixed << std::setprecision(2)
                  << blockidx_vs_smid_diff << " GB/s (" << (blockidx_vs_smid_diff > 0 ? "+" : "") << std::setprecision(2) << blockidx_vs_smid_pct << "%)" << std::endl;

        double row_vs_smid_diff = blockidx_row_bw - smid_bw;
        double row_vs_smid_pct = (row_vs_smid_diff / smid_bw) * 100.0;
        std::cout << "  BlockIdx-row vs SMID: " << (row_vs_smid_diff > 0 ? "+" : "") << std::fixed << std::setprecision(2)
                  << row_vs_smid_diff << " GB/s (" << (row_vs_smid_diff > 0 ? "+" : "") << std::setprecision(2) << row_vs_smid_pct << "%)" << std::endl;

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Y));
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "测试完成!" << std::endl;
    std::cout << "========================================================" << std::endl;

    return 0;
}
