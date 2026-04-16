#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// 检查 CUDA 错误的宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// =========================================================
// Kernel 1: 基于 smid 控制的 Element-wise
// =========================================================
__global__ void elementwise_smid_based(const float* __restrict__ A,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       size_t N, int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    // 如果当前硬件分配的 SM ID 超过了限制，直接退出
    if (my_smid >= sm_count) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = sm_count * warps_per_block;

    size_t stride = total_active_warps * 32;
    size_t start_idx = global_warpID * 32 + laneID;

    // 跨步幅遍历整个一维数组
    for (size_t i = start_idx; i < N; i += stride) {
        Y[i] = A[i] + X[i];
    }
}

// =========================================================
// Kernel 2: 基于 warpID 控制的 Element-wise
// =========================================================
__global__ void elementwise_warpid_based(const float* __restrict__ A,
                                         const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         size_t N, int active_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = tid / 32;
    int laneID = threadIdx.x % 32;

    int warps_per_block = blockDim.x / 32;
    int max_active_warps = active_blocks * warps_per_block;

    // 如果当前分配到的全局 Warp ID 超过了我们设定的工作量上限，直接退出
    if (warpID >= max_active_warps) return;

    size_t stride = max_active_warps * 32;
    size_t start_idx = warpID * 32 + laneID;

    // 跨步幅遍历整个一维数组
    for (size_t i = start_idx; i < N; i += stride) {
        Y[i] = A[i] + X[i];
    }
}

// =========================================================
// CPU 参考实现与验证逻辑
// =========================================================
void elementwise_cpu_reference(const float* A, const float* X, float* Y, size_t N) {
    for (size_t i = 0; i < N; i++) {
        Y[i] = A[i] + X[i];
    }
}

bool verify_results(const float* expected, const float* actual, size_t N, float tolerance = 1e-4f) {
    for (size_t i = 0; i < N; i++) {
        float diff = std::abs(expected[i] - actual[i]);
        float rel_error = diff / (std::abs(expected[i]) + 1e-8f);
        if (rel_error > tolerance && diff > tolerance) {
            // 只打印前几个错误，避免刷屏
            std::cerr << "验证失败：位置 " << i << ", 期望=" << expected[i]
                      << ", 实际=" << actual[i] << ", 相对误差=" << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

enum KernelType { SMID_BASED, WARPID_BASED };

float test_elementwise_performance(KernelType type, const float* d_A, const float* d_X, float* d_Y,
                                   size_t N, int blocks_limit, int threads_per_block) {
    // 强制启动 132 个 Block 铺满调度器
    int launch_blocks = 132;

    // 预热
    if (type == SMID_BASED) {
        elementwise_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
    } else {
        elementwise_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (type == SMID_BASED) {
            elementwise_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
        } else {
            elementwise_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
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

bool verify_kernel(KernelType type, const float* d_A, const float* d_X, float* d_Y,
                   size_t N, int blocks_limit, int threads_per_block,
                   const float* h_Y_cpu) {
    
    // 初始化设备端 Y 为全 0，防止上一轮残留数据掩盖漏算
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    int launch_blocks = 132;
    if (type == SMID_BASED) {
        elementwise_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
    } else {
        elementwise_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_Y_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_Y_gpu.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    return verify_results(h_Y_cpu, h_Y_gpu.data(), N);
}

int main() {
    // 数组大小：1千万到2亿个 float 元素
    std::vector<size_t> test_sizes = {
        2000000,
        5000000,
        10000000,   // ~114 MB (小数据)
        50000000,   // ~572 MB 
        100000000,  // ~1.14 GB (大数据，推荐)
        200000000   // ~2.28 GB
    };

    std::vector<int> block_counts = {16, 33, 66, 108, 132};
    int threads_per_block = 256;

    for (size_t N : test_sizes) {
        size_t size_bytes = N * sizeof(float);

        std::vector<float> h_A(N, 1.0f);
        std::vector<float> h_X(N, 2.0f);
        std::vector<float> h_Y_cpu(N, 0.0f);
        
        // 预先算好 CPU 参考结果
        elementwise_cpu_reference(h_A.data(), h_X.data(), h_Y_cpu.data(), N);

        float *d_A, *d_X, *d_Y;
        CHECK_CUDA(cudaMalloc(&d_A, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_X, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_Y, size_bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_bytes, cudaMemcpyHostToDevice));

        // 每次读取 A 和 X (2 读)，写入 Y (1 写)
        double total_rw_bytes = (double)N * 3.0 * sizeof(float);
        double total_gb = total_rw_bytes / (1024.0 * 1024.0 * 1024.0);

        std::cout << "\n========================================================" << std::endl;
        std::cout << "测试元素总量：" << N << " (FP32)" << std::endl;
        std::cout << "单次理论访存：~" << total_gb << " GB" << std::endl;
        std::cout << "========================================================\n" << std::endl;

        std::cout << "正确性验证 (132 blocks 满载):" << std::endl;
        bool smid_pass = verify_kernel(SMID_BASED, d_A, d_X, d_Y, N, 132, threads_per_block, h_Y_cpu.data());
        bool warpid_pass = verify_kernel(WARPID_BASED, d_A, d_X, d_Y, N, 132, threads_per_block, h_Y_cpu.data());

        std::cout << "  SMID_BASED:   " << (smid_pass ? "PASS" : "FAIL (通常由于硬件 SM ID 不连续导致漏算)") << std::endl;
        std::cout << "  WARPID_BASED: " << (warpid_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;

        std::cout << std::left << std::setw(15) << "Blocks_Limit"
                  << std::setw(25) << "SMID_Based (GB/s)"
                  << std::setw(25) << "WarpID_Based (GB/s)" << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;

        for (int blocks : block_counts) {
            float ms_smid = test_elementwise_performance(SMID_BASED, d_A, d_X, d_Y, N, blocks, threads_per_block);
            double bw_smid = (total_rw_bytes / (ms_smid / 1000.0)) / 1e9;

            float ms_warpid = test_elementwise_performance(WARPID_BASED, d_A, d_X, d_Y, N, blocks, threads_per_block);
            double bw_warpid = (total_rw_bytes / (ms_warpid / 1000.0)) / 1e9;

            std::cout << std::left << std::setw(15) << blocks
                      << std::setw(25) << std::to_string((int)(bw_smid * 100) / 100.0)
                      << std::setw(25) << std::to_string((int)(bw_warpid * 100) / 100.0)
                      << std::endl;
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Y));
    }

    return 0;
}