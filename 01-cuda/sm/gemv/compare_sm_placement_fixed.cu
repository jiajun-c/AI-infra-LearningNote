#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

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

// =====================================================================
// Kernel 1: WarpID / BlockIdx 限制版 (官方推荐写法)
// 逻辑: 只有前 66 个 Block 存活，完美规避 SMID 的不连续和调度分配问题
// =====================================================================
__global__ void gemv_warpid_66sm(const float* __restrict__ A,
                                 const float* __restrict__ X,
                                 float* __restrict__ Y,
                                 int N, int M, int target_sms,
                                 int* d_row_counter) {
    // 假设 1 个 Block 霸占 1 个 SM。超出 target_sms 的 Block 直接自杀
    if (blockIdx.x >= target_sms) return;

    int laneID = threadIdx.x % 32;

    // Warp 级别的原子抢占任务
    while (true) {
        int row = 0;
        if (laneID == 0) {
            row = atomicAdd(d_row_counter, 1);
        }
        // 广播给当前 Warp 内的所有 32 个线程
        row = __shfl_sync(0xffffffff, row, 0);

        if (row >= N) break;

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

// =====================================================================
// Kernel 2: 连续 SMID 限制版 (强制使用 SM 0 ~ 65)
// 逻辑: 依赖底层汇编，必须配有 Atomic 计数器来防止硬件分配不均导致的漏算
// =====================================================================
__global__ void gemv_smid_sequential_66sm(const float* __restrict__ A,
                                          const float* __restrict__ X,
                                          float* __restrict__ Y,
                                          int N, int M, int target_sms,
                                          int* d_row_counter) {
    unsigned int my_smid = get_smid();
    
    // 严格限制：只有硬件 SM_ID 在 0 到 65 之间的才能活下来干活
    if (my_smid >= target_sms) return;

    int laneID = threadIdx.x % 32;

    while (true) {
        int row = 0;
        if (laneID == 0) {
            row = atomicAdd(d_row_counter, 1);
        }
        row = __shfl_sync(0xffffffff, row, 0);

        if (row >= N) break;

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

// =====================================================================
// Kernel 3: 交错 SMID 限制版 (强制使用 SM 0, 2, 4...130)
// 逻辑: 试图在全芯片分散热量和 L2 缓存压力，严格占用 66 个 SM
// =====================================================================
__global__ void gemv_smid_interleaved_66sm(const float* __restrict__ A,
                                           const float* __restrict__ X,
                                           float* __restrict__ Y,
                                           int N, int M, 
                                           int* d_row_counter) {
    unsigned int my_smid = get_smid();
    
    // 严格限制：只能是偶数 SMID 且不能越界，总计 66 个槽位
    if (my_smid % 2 != 0 || my_smid >= 132) return;

    int laneID = threadIdx.x % 32;

    while (true) {
        int row = 0;
        if (laneID == 0) {
            row = atomicAdd(d_row_counter, 1);
        }
        row = __shfl_sync(0xffffffff, row, 0);

        if (row >= N) break;

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

// =====================================================================
// CPU 参考实现与验证逻辑
// =====================================================================
void gemv_cpu_reference(const float* A, const float* X, float* Y, int N, int M) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            sum += A[i * M + j] * X[j];
        }
        Y[i] = sum;
    }
}

bool verify_results(const float* expected, const float* actual, int N) {
    for (int i = 0; i < N; i++) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > 1e-3f) { // 考虑浮点数累加的精度损失
            std::cerr << "验证失败 @ row " << i << " : 期望=" << expected[i] << " 实际=" << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

// =====================================================================
// 性能测试封装
// =====================================================================
enum KernelType { WARPID_66, SMID_SEQ_66, SMID_INTER_66 };

float test_kernel(KernelType type, const float* d_A, const float* d_X, float* d_Y, 
                  int N, int M, int* d_row_counter) {
    
    // 统一发射 132 个 Block，把硬件铺满，然后在 Kernel 内部“杀掉”一半
    int launch_blocks = 132; 
    int threads_per_block = 256; 
    int target_sms = 66;

    // 每次执行前，必须清零原子计数器和结果数组
    CHECK_CUDA(cudaMemset(d_row_counter, 0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    // 预热 (Warmup)
    for (int i = 0; i < 5; i++) {
        CHECK_CUDA(cudaMemset(d_row_counter, 0, sizeof(int)));
        if (type == WARPID_66) {
            gemv_warpid_66sm<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, target_sms, d_row_counter);
        } else if (type == SMID_SEQ_66) {
            gemv_smid_sequential_66sm<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, target_sms, d_row_counter);
        } else if (type == SMID_INTER_66) {
            gemv_smid_interleaved_66sm<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, d_row_counter);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100; // 增加循环次数，消除误差
    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        CHECK_CUDA(cudaMemset(d_row_counter, 0, sizeof(int))); // 极其重要，每次都要重置行号
        
        if (type == WARPID_66) {
            gemv_warpid_66sm<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, target_sms, d_row_counter);
        } else if (type == SMID_SEQ_66) {
            gemv_smid_sequential_66sm<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, target_sms, d_row_counter);
        } else if (type == SMID_INTER_66) {
            gemv_smid_interleaved_66sm<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, d_row_counter);
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

int main() {
    // 使用能撑爆带宽的大矩阵
    const int N = 16384;
    const int M = 16384;

    size_t size_A = N * M * sizeof(float);
    size_t size_X = M * sizeof(float);
    size_t size_Y = N * sizeof(float);

    std::vector<float> h_A(N * M, 1.0f); // 简单填充 1.0 方便验证
    std::vector<float> h_X(M, 1.0f);
    std::vector<float> h_Y_cpu(N, 0.0f);
    std::vector<float> h_Y_gpu(N, 0.0f);

    std::cout << "正在计算 CPU 参考结果..." << std::endl;
    gemv_cpu_reference(h_A.data(), h_X.data(), h_Y_cpu.data(), N, M);

    float *d_A, *d_X, *d_Y;
    int *d_row_counter; // 原子计数器指针
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_X, size_X));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y));
    CHECK_CUDA(cudaMalloc(&d_row_counter, sizeof(int))); // 分配原子计数器内存

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice));

    double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;
    double total_bytes = bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << "\n========================================================" << std::endl;
    std::cout << "测试矩阵大小: " << N << " x " << M << " (FP32)" << std::endl;
    std::cout << "理论访存量  : ~" << std::fixed << std::setprecision(4) << total_bytes << " GB" << std::endl;
    std::cout << "约束条件    : 强行锁死 66 个 SM" << std::endl;
    std::cout << "========================================================\n" << std::endl;

    std::cout << std::left << std::setw(25) << "策略 (仅66个SM工作)" 
              << std::setw(15) << "正确性" 
              << std::setw(15) << "平均耗时" 
              << std::setw(20) << "显存带宽 (GB/s)" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;

    KernelType types[] = {WARPID_66, SMID_SEQ_66, SMID_INTER_66};
    const char* names[] = {"WarpID 限制 (推荐)", "SMID 连续 (0~65)", "SMID 交错 (偶数)"};

    for (int i = 0; i < 3; i++) {
        // 1. 测速
        float ms = test_kernel(types[i], d_A, d_X, d_Y, N, M, d_row_counter);
        
        // 2. 拷贝结果并验证正确性
        CHECK_CUDA(cudaMemcpy(h_Y_gpu.data(), d_Y, size_Y, cudaMemcpyDeviceToHost));
        bool is_correct = verify_results(h_Y_cpu.data(), h_Y_gpu.data(), N);
        
        double bw = (total_bytes * 1024.0 * 1024.0 * 1024.0) / (ms * 1e6); // GB/s 换算

        std::cout << std::left << std::setw(25) << names[i] 
                  << std::setw(15) << (is_correct ? "PASS" : "FAIL") 
                  << std::fixed << std::setprecision(3) << ms << " ms       "
                  << std::fixed << std::setprecision(2) << bw << std::endl;
    }
    std::cout << "========================================================================" << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_row_counter));

    return 0;
}