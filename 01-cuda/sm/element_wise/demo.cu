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

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// =========================================================
// Kernel 1: 基于 smid 控制的 Element-wise (Add only)
// =========================================================
__global__ void elementwise_smid_based(const float* __restrict__ A,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       size_t N, int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    if (my_smid >= sm_count) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = sm_count * warps_per_block;

    size_t stride = total_active_warps * 32;
    size_t start_idx = global_warpID * 32 + laneID;

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

    if (warpID >= max_active_warps) return;

    size_t stride = max_active_warps * 32;
    size_t start_idx = warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        Y[i] = A[i] + X[i];
    }
}

// =========================================================
// Kernel 3: Fused Add + ReLU + Multiply (SMID-based)
// Y = ReLU(A + X) * Scale
// =========================================================
__global__ void fused_add_relu_mul_smid(const float* __restrict__ A,
                                        const float* __restrict__ X,
                                        const float* __restrict__ Scale,
                                        float* __restrict__ Y,
                                        size_t N, int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    if (my_smid >= sm_count) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = sm_count * warps_per_block;

    size_t stride = total_active_warps * 32;
    size_t start_idx = global_warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        float sum = A[i] + X[i];
        float after_relu = relu(sum);
        Y[i] = after_relu * Scale[i];
    }
}

// =========================================================
// Kernel 4: Fused Add + ReLU + Multiply (WarpID-based)
// =========================================================
__global__ void fused_add_relu_mul_warpid(const float* __restrict__ A,
                                          const float* __restrict__ X,
                                          const float* __restrict__ Scale,
                                          float* __restrict__ Y,
                                          size_t N, int active_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = tid / 32;
    int laneID = threadIdx.x % 32;

    int warps_per_block = blockDim.x / 32;
    int max_active_warps = active_blocks * warps_per_block;

    if (warpID >= max_active_warps) return;

    size_t stride = max_active_warps * 32;
    size_t start_idx = warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        float sum = A[i] + X[i];
        float after_relu = relu(sum);
        Y[i] = after_relu * Scale[i];
    }
}

// =========================================================
// Kernel 5: Separate Add kernel
// =========================================================
__global__ void separate_add(const float* __restrict__ A,
                             const float* __restrict__ X,
                             float* __restrict__ Y,
                             size_t N, int active_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = tid / 32;
    int laneID = threadIdx.x % 32;

    int warps_per_block = blockDim.x / 32;
    int max_active_warps = active_blocks * warps_per_block;

    if (warpID >= max_active_warps) return;

    size_t stride = max_active_warps * 32;
    size_t start_idx = warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        Y[i] = A[i] + X[i];
    }
}

// =========================================================
// Kernel 6: Separate ReLU + Multiply kernel (WarpID-based)
// =========================================================
__global__ void separate_relu_mul(const float* __restrict__ In,
                                  const float* __restrict__ Scale,
                                  float* __restrict__ Out,
                                  size_t N, int active_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = tid / 32;
    int laneID = threadIdx.x % 32;

    int warps_per_block = blockDim.x / 32;
    int max_active_warps = active_blocks * warps_per_block;

    if (warpID >= max_active_warps) return;

    size_t stride = max_active_warps * 32;
    size_t start_idx = warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        float after_relu = relu(In[i]);
        Out[i] = after_relu * Scale[i];
    }
}

// =========================================================
// Kernel 7: Separate Add kernel (SMID-based)
// =========================================================
__global__ void separate_add_smid(const float* __restrict__ A,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y,
                                  size_t N, int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    if (my_smid >= sm_count) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = sm_count * warps_per_block;

    size_t stride = total_active_warps * 32;
    size_t start_idx = global_warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        Y[i] = A[i] + X[i];
    }
}

// =========================================================
// Kernel 8: Separate ReLU + Multiply kernel (SMID-based)
// =========================================================
__global__ void separate_relu_mul_smid(const float* __restrict__ In,
                                       const float* __restrict__ Scale,
                                       float* __restrict__ Out,
                                       size_t N, int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    if (my_smid >= sm_count) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = sm_count * warps_per_block;

    size_t stride = total_active_warps * 32;
    size_t start_idx = global_warpID * 32 + laneID;

    for (size_t i = start_idx; i < N; i += stride) {
        float after_relu = relu(In[i]);
        Out[i] = after_relu * Scale[i];
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

void fused_cpu_reference(const float* A, const float* X, const float* Scale, float* Y, size_t N) {
    for (size_t i = 0; i < N; i++) {
        float sum = A[i] + X[i];
        float after_relu = fmaxf(0.0f, sum);
        Y[i] = after_relu * Scale[i];
    }
}

bool verify_results(const float* expected, const float* actual, size_t N, float tolerance = 1e-4f) {
    for (size_t i = 0; i < N; i++) {
        float diff = std::abs(expected[i] - actual[i]);
        float rel_error = diff / (std::abs(expected[i]) + 1e-8f);
        if (rel_error > tolerance && diff > tolerance) {
            std::cerr << "验证失败：位置 " << i << ", 期望=" << expected[i]
                      << ", 实际=" << actual[i] << ", 相对误差=" << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

enum KernelType { SMID_BASED, WARPID_BASED, FUSED_SMID, FUSED_WARPID, SEPARATE_SMID, SEPARATE_WARPID };

float test_fused_separate(KernelType type, const float* d_A, const float* d_X,
                          const float* d_Scale, float* d_Y, float* d_Tmp,
                          size_t N, int blocks_limit, int threads_per_block) {
    int launch_blocks = 132;

    if (type == SEPARATE_SMID) {
        // Separate: Add -> ReLU+Mul 两个 kernel (SMID 版本)
        separate_add_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Tmp, N, blocks_limit);
        separate_relu_mul_smid<<<launch_blocks, threads_per_block>>>(d_Tmp, d_Scale, d_Y, N, blocks_limit);
    } else if (type == SEPARATE_WARPID) {
        // Separate: Add -> ReLU+Mul 两个 kernel (WarpID 版本)
        separate_add<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Tmp, N, blocks_limit);
        separate_relu_mul<<<launch_blocks, threads_per_block>>>(d_Tmp, d_Scale, d_Y, N, blocks_limit);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (type == SEPARATE_SMID) {
            separate_add_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Tmp, N, blocks_limit);
            separate_relu_mul_smid<<<launch_blocks, threads_per_block>>>(d_Tmp, d_Scale, d_Y, N, blocks_limit);
        } else if (type == SEPARATE_WARPID) {
            separate_add<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Tmp, N, blocks_limit);
            separate_relu_mul<<<launch_blocks, threads_per_block>>>(d_Tmp, d_Scale, d_Y, N, blocks_limit);
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

float test_elementwise_performance(KernelType type, const float* d_A, const float* d_X, float* d_Y,
                                   size_t N, int blocks_limit, int threads_per_block) {
    int launch_blocks = 132;

    if (type == SMID_BASED) {
        elementwise_smid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
    } else if (type == WARPID_BASED) {
        elementwise_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, blocks_limit);
    } else if (type == FUSED_SMID) {
        // Fused 版本需要额外的 Scale 参数，这里不使用此函数
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
        } else if (type == WARPID_BASED) {
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

float test_fused_performance(KernelType type, const float* d_A, const float* d_X,
                             const float* d_Scale, float* d_Y, float* d_Tmp,
                             size_t N, int blocks_limit, int threads_per_block) {
    int launch_blocks = 132;

    if (type == FUSED_SMID) {
        fused_add_relu_mul_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Scale, d_Y, N, blocks_limit);
    } else if (type == FUSED_WARPID) {
        fused_add_relu_mul_warpid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Scale, d_Y, N, blocks_limit);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (type == FUSED_SMID) {
            fused_add_relu_mul_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Scale, d_Y, N, blocks_limit);
        } else if (type == FUSED_WARPID) {
            fused_add_relu_mul_warpid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Scale, d_Y, N, blocks_limit);
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

bool verify_fused_kernel(const float* d_A, const float* d_X, const float* d_Scale, float* d_Y,
                         size_t N, int blocks_limit, int threads_per_block,
                         const float* h_Y_cpu) {
    CHECK_CUDA(cudaMemset(d_Y, 0, N * sizeof(float)));

    int launch_blocks = 132;
    fused_add_relu_mul_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Scale, d_Y, N, blocks_limit);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_Y_gpu(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_Y_gpu.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    return verify_results(h_Y_cpu, h_Y_gpu.data(), N);
}

int main() {
    std::vector<size_t> test_sizes = {
        10000000,   // ~114 MB
        50000000,   // ~572 MB
        100000000,  // ~1.14 GB
        200000000   // ~2.28 GB
    };

    std::vector<int> block_counts = {16, 33, 66, 108, 132};
    int threads_per_block = 256;

    std::cout << "============================================================" << std::endl;
    std::cout << "实验 1: 简单 Add 操作 - SMID vs WarpID" << std::endl;
    std::cout << "============================================================" << std::endl;

    for (size_t N : test_sizes) {
        size_t size_bytes = N * sizeof(float);

        std::vector<float> h_A(N, 1.0f);
        std::vector<float> h_X(N, 2.0f);
        std::vector<float> h_Y_cpu(N, 0.0f);

        elementwise_cpu_reference(h_A.data(), h_X.data(), h_Y_cpu.data(), N);

        float *d_A, *d_X, *d_Y;
        CHECK_CUDA(cudaMalloc(&d_A, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_X, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_Y, size_bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_bytes, cudaMemcpyHostToDevice));

        double total_rw_bytes = (double)N * 3.0 * sizeof(float);
        double total_gb = total_rw_bytes / (1024.0 * 1024.0 * 1024.0);

        std::cout << "\n--------------------------------------------------------" << std::endl;
        std::cout << "测试元素总量：" << N << " (FP32), 访存：" << total_gb << " GB" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;

        std::cout << std::left << std::setw(15) << "Blocks"
                  << std::setw(20) << "SMID (GB/s)"
                  << std::setw(20) << "WarpID (GB/s)" << std::endl;

        for (int blocks : block_counts) {
            float ms_smid = test_elementwise_performance(SMID_BASED, d_A, d_X, d_Y, N, blocks, threads_per_block);
            double bw_smid = (total_rw_bytes / (ms_smid / 1000.0)) / 1e9;

            float ms_warpid = test_elementwise_performance(WARPID_BASED, d_A, d_X, d_Y, N, blocks, threads_per_block);
            double bw_warpid = (total_rw_bytes / (ms_warpid / 1000.0)) / 1e9;

            std::cout << std::left << std::setw(15) << blocks
                      << std::setw(20) << std::to_string((int)(bw_smid * 100) / 100.0)
                      << std::setw(20) << std::to_string((int)(bw_warpid * 100) / 100.0)
                      << std::endl;
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Y));
    }

    std::cout << "\n\n============================================================" << std::endl;
    std::cout << "实验 2: Fused Add+ReLU+Mul vs Separate (两个 kernel)" << std::endl;
    std::cout << "Fused: Y = ReLU(A + X) * Scale (单 kernel)" << std::endl;
    std::cout << "Separate: Add -> ReLU+Mul (双 kernel)" << std::endl;
    std::cout << "============================================================" << std::endl;

    for (size_t N : test_sizes) {
        size_t size_bytes = N * sizeof(float);

        std::vector<float> h_A(N, 1.0f);
        std::vector<float> h_X(N, 2.0f);
        std::vector<float> h_Scale(N, 0.5f);
        std::vector<float> h_Y_cpu(N, 0.0f);

        fused_cpu_reference(h_A.data(), h_X.data(), h_Scale.data(), h_Y_cpu.data(), N);

        float *d_A, *d_X, *d_Scale, *d_Y, *d_Tmp;
        CHECK_CUDA(cudaMalloc(&d_A, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_X, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_Scale, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_Y, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_Tmp, size_bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Scale, h_Scale.data(), size_bytes, cudaMemcpyHostToDevice));

        // Fused: 3 读 (A, X, Scale) + 1 写 (Y) = 4 次内存访问
        double fused_rw_bytes = (double)N * 4.0 * sizeof(float);
        // Separate: Add(2 读 1 写) + ReLU+Mul(2 读 1 写) = 6 次内存访问
        double separate_rw_bytes = (double)N * 6.0 * sizeof(float);

        std::cout << "\n--------------------------------------------------------" << std::endl;
        std::cout << "测试元素总量：" << N << " (FP32)" << std::endl;
        std::cout << "Fused 访存：~" << fused_rw_bytes / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        std::cout << "Separate 访存：~" << separate_rw_bytes / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;

        // 正确性验证
        bool fused_pass = verify_fused_kernel(d_A, d_X, d_Scale, d_Y, N, 132, threads_per_block, h_Y_cpu.data());
        std::cout << "Fused Kernel 验证：" << (fused_pass ? "PASS" : "FAIL") << std::endl << std::endl;

        std::cout << std::left << std::setw(15) << "Blocks"
                  << std::setw(18) << "Fused SMID"
                  << std::setw(18) << "Fused WarpID"
                  << std::setw(18) << "Sep SMID"
                  << std::setw(18) << "Sep WarpID"
                  << std::setw(15) << "Fused 提升" << std::endl;

        for (int blocks : block_counts) {
            float ms_fused_smid = test_fused_performance(FUSED_SMID, d_A, d_X, d_Scale, d_Y, d_Tmp, N, blocks, threads_per_block);
            float ms_fused_warpid = test_fused_performance(FUSED_WARPID, d_A, d_X, d_Scale, d_Y, d_Tmp, N, blocks, threads_per_block);

            // Separate 版本也需要 SMID 和 WarpID 两种 kernel
            float ms_sep_smid = test_fused_separate(SEPARATE_SMID, d_A, d_X, d_Scale, d_Y, d_Tmp, N, blocks, threads_per_block);
            float ms_sep_warpid = test_fused_separate(SEPARATE_WARPID, d_A, d_X, d_Scale, d_Y, d_Tmp, N, blocks, threads_per_block);

            double bw_fused_smid = (fused_rw_bytes / (ms_fused_smid / 1000.0)) / 1e9;
            double bw_fused_warpid = (fused_rw_bytes / (ms_fused_warpid / 1000.0)) / 1e9;
            double bw_sep_smid = (separate_rw_bytes / (ms_sep_smid / 1000.0)) / 1e9;
            double bw_sep_warpid = (separate_rw_bytes / (ms_sep_warpid / 1000.0)) / 1e9;

            // 计算速度提升 (time-based)
            float speedup_smid = (ms_sep_smid - ms_fused_smid) / ms_sep_smid * 100;
            float speedup_warpid = (ms_sep_warpid - ms_fused_warpid) / ms_sep_warpid * 100;

            std::cout << std::left << std::setw(15) << blocks
                      << std::setw(18) << std::to_string((int)(bw_fused_smid * 100) / 100.0)
                      << std::setw(18) << std::to_string((int)(bw_fused_warpid * 100) / 100.0)
                      << std::setw(18) << std::to_string((int)(bw_sep_smid * 100) / 100.0)
                      << std::setw(18) << std::to_string((int)(bw_sep_warpid * 100) / 100.0)
                      << std::setw(15) << ("SM:" + std::to_string((int)(speedup_smid * 100) / 100.0) + "% W:" + std::to_string((int)(speedup_warpid * 100) / 100.0) + "%")
                      << std::endl;
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_X));
        CHECK_CUDA(cudaFree(d_Scale));
        CHECK_CUDA(cudaFree(d_Y));
        CHECK_CUDA(cudaFree(d_Tmp));
    }

    return 0;
}
