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

// Kernel 1: 顺序 SM 放置 (Sequential SM placement)
// SM 0, 1, 2, ..., sm_count-1 被使用
__global__ void gemv_sequential_smid(const float* __restrict__ A,
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

// Kernel 2: 交错 SM 放置 (Interleaved SM placement)
// 使用 SM 0, 2, 4, ... (每隔一个 SM) 或使用某种交错模式
// 这样可以更好地分散 L2 cache 访问
__global__ void gemv_interleaved_smid(const float* __restrict__ A,
                                      const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      int N, int M, int total_sms, int target_sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    // 交错模式：只使用偶数编号的 SM (0, 2, 4, ...)
    // 或者使用 stride 模式来分散 SM
    int stride = 2;  // 每隔一个 SM
    int max_target_sms = total_sms / stride;

    // 只使用符合交错模式的 SM
    if (my_smid % stride != 0) return;

    int target_sm_idx = my_smid / stride;
    if (target_sm_idx >= target_sm_count) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = target_sm_idx * warps_per_block + local_warpID;
    int total_active_warps = target_sm_count * warps_per_block;

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

// Kernel 3: 交错 SM 放置 - 可配置 stride
__global__ void gemv_interleaved_smid_stride(const float* __restrict__ A,
                                             const float* __restrict__ X,
                                             float* __restrict__ Y,
                                             int N, int M, int total_sms, int stride, int start_sm) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    // 交错模式：使用 SM start_sm, start_sm+stride, start_sm+2*stride, ...
    if ((my_smid - start_sm + stride) % stride != 0) return;

    int target_sm_idx = (my_smid - start_sm + stride) % total_sms / stride;
    int max_sms_for_stride = (total_sms - start_sm + stride - 1) / stride;
    if (target_sm_idx >= max_sms_for_stride) return;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = target_sm_idx * warps_per_block + local_warpID;
    int total_active_warps = max_sms_for_stride * warps_per_block;

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

// Kernel 4: WarpID 版本 (原始)
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

// Kernel 5: Standard 版本
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

enum KernelType { SEQUENTIAL_SMID, INTERLEAVED_SMID, INTERLEAVED_STRIDE, WARPID_BASED, STANDARD, CUBLAS };

void print_sm_distribution(KernelType type, int total_sms, int target_count, int stride = 2, int start_sm = 0) {
    std::cout << "SM 分布：";
    if (type == SEQUENTIAL_SMID) {
        std::cout << "SM 0-" << (target_count - 1) << " (连续)";
    } else if (type == INTERLEAVED_SMID) {
        std::cout << "SM 0,2,4,...,2*" << (target_count - 1) << " (交错 stride=2)";
    } else if (type == INTERLEAVED_STRIDE) {
        std::cout << "SM " << start_sm << "," << (start_sm + stride) << "," << (start_sm + 2 * stride) << ",... (交错 stride=" << stride << ")";
    } else if (type == WARPID_BASED || type == STANDARD) {
        std::cout << "由调度器决定";
    } else if (type == CUBLAS) {
        std::cout << "cuBLAS 内部决定";
    }
    std::cout << std::endl;
}

float test_gemv_performance(KernelType type, const float* d_A, const float* d_X, float* d_Y,
                            int N, int M, int blocks_limit, int threads_per_block,
                            cublasHandle_t cublas_handle, int total_sms, int stride = 2, int start_sm = 0) {
    int launch_blocks = blocks_limit;
    float alpha = 1.0f, beta = 0.0f;

    if (type == CUBLAS) {
        CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N,
                                  &alpha, d_A, M, d_X, 1, &beta, d_Y, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (type == SEQUENTIAL_SMID) {
        gemv_sequential_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (type == INTERLEAVED_SMID) {
        gemv_interleaved_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, total_sms, blocks_limit);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (type == INTERLEAVED_STRIDE) {
        gemv_interleaved_smid_stride<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, total_sms, stride, start_sm);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (type == WARPID_BASED) {
        gemv_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else if (type == STANDARD) {
        gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Warmup 20 次
    for (int i = 0; i < 20; i++) {
        if (type == CUBLAS) {
            cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
        } else if (type == SEQUENTIAL_SMID) {
            gemv_sequential_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == INTERLEAVED_SMID) {
            gemv_interleaved_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, total_sms, blocks_limit);
        } else if (type == INTERLEAVED_STRIDE) {
            gemv_interleaved_smid_stride<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, total_sms, stride, start_sm);
        } else if (type == WARPID_BASED) {
            gemv_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == STANDARD) {
            gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
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
        } else if (type == SEQUENTIAL_SMID) {
            gemv_sequential_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == INTERLEAVED_SMID) {
            gemv_interleaved_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, total_sms, blocks_limit);
        } else if (type == INTERLEAVED_STRIDE) {
            gemv_interleaved_smid_stride<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, total_sms, stride, start_sm);
        } else if (type == WARPID_BASED) {
            gemv_warpid_based<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M, blocks_limit);
        } else if (type == STANDARD) {
            gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X, d_Y, N, M);
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
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    int total_sms = prop.multiProcessorCount;

    std::cout << "Device: " << prop.name << " | Total SMs: " << total_sms << std::endl;
    std::cout << "========================================================" << std::endl;

    const int N = 512;
    const int M = 1024;
    const int blocks_limit = 132;
    const int threads_per_block = 256;

    size_t size_A = N * M * sizeof(float);
    size_t size_X = M * sizeof(float);
    size_t size_Y = N * sizeof(float);

    double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;
    double total_bytes = bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << "测试矩阵大小：" << N << " x " << M << " (FP32)" << std::endl;
    std::cout << "访存量：~" << std::fixed << std::setprecision(6) << total_bytes << " GB" << std::endl;
    std::cout << "Blocks: " << blocks_limit << ", Threads/block: " << threads_per_block << std::endl;
    std::cout << "========================================================" << std::endl;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float *d_A, *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_X, size_X));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y));

    std::vector<float> h_A(N * M, 1.0f);
    std::vector<float> h_X(M, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice));

    std::cout << "\n各版本性能对比 (warmup=20, runs=2):" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 1. Sequential SMID
    std::cout << "\n1. Sequential SMID (连续 SM 放置):" << std::endl;
    print_sm_distribution(SEQUENTIAL_SMID, total_sms, blocks_limit);
    float avg_ms_seq = test_gemv_performance(SEQUENTIAL_SMID, d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle, total_sms);
    double bw_seq = (bytes / (avg_ms_seq / 1000.0)) / 1e9;
    std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms_seq << " ms" << std::endl;
    std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bw_seq << " GB/s" << std::endl;

    // 2. Interleaved SMID (stride=2)
    std::cout << "\n2. Interleaved SMID (交错 SM 放置，stride=2):" << std::endl;
    print_sm_distribution(INTERLEAVED_SMID, total_sms, blocks_limit);
    float avg_ms_inter = test_gemv_performance(INTERLEAVED_SMID, d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle, total_sms);
    double bw_inter = (bytes / (avg_ms_inter / 1000.0)) / 1e9;
    std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms_inter << " ms" << std::endl;
    std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bw_inter << " GB/s" << std::endl;

    // 3. Interleaved SMID with configurable stride
    int stride = 2;
    int start_sm = 0;
    std::cout << "\n3. Interleaved SMID (可配置 stride=" << stride << ", start=" << start_sm << "):" << std::endl;
    print_sm_distribution(INTERLEAVED_STRIDE, total_sms, blocks_limit, stride, start_sm);
    float avg_ms_stride = test_gemv_performance(INTERLEAVED_STRIDE, d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle, total_sms, stride, start_sm);
    double bw_stride = (bytes / (avg_ms_stride / 1000.0)) / 1e9;
    std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms_stride << " ms" << std::endl;
    std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bw_stride << " GB/s" << std::endl;

    // 4. WarpID based
    std::cout << "\n4. WarpID based (原始):" << std::endl;
    print_sm_distribution(WARPID_BASED, total_sms, blocks_limit);
    float avg_ms_warp = test_gemv_performance(WARPID_BASED, d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle, total_sms);
    double bw_warp = (bytes / (avg_ms_warp / 1000.0)) / 1e9;
    std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms_warp << " ms" << std::endl;
    std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bw_warp << " GB/s" << std::endl;

    // 5. Standard
    std::cout << "\n5. Standard (blocks 内部分配):" << std::endl;
    print_sm_distribution(STANDARD, total_sms, blocks_limit);
    float avg_ms_std = test_gemv_performance(STANDARD, d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle, total_sms);
    double bw_std = (bytes / (avg_ms_std / 1000.0)) / 1e9;
    std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms_std << " ms" << std::endl;
    std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bw_std << " GB/s" << std::endl;

    // 6. cuBLAS
    std::cout << "\n6. cuBLAS:" << std::endl;
    print_sm_distribution(CUBLAS, total_sms, blocks_limit);
    float avg_ms_cublas = test_gemv_performance(CUBLAS, d_A, d_X, d_Y, N, M, blocks_limit, threads_per_block, cublas_handle, total_sms);
    double bw_cublas = (bytes / (avg_ms_cublas / 1000.0)) / 1e9;
    std::cout << "  平均时间：" << std::fixed << std::setprecision(6) << avg_ms_cublas << " ms" << std::endl;
    std::cout << "  带宽：" << std::fixed << std::setprecision(2) << bw_cublas << " GB/s" << std::endl;

    // 汇总对比
    std::cout << "\n========================================================" << std::endl;
    std::cout << "性能汇总:" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(35) << "Kernel" << std::setw(15) << "Time (ms)" << std::setw(18) << "Bandwidth (GB/s)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(35) << "Sequential SMID" << std::setw(15) << avg_ms_seq << std::setw(18) << bw_seq << std::endl;
    std::cout << std::left << std::setw(35) << "Interleaved SMID" << std::setw(15) << avg_ms_inter << std::setw(18) << bw_inter << std::endl;
    std::cout << std::left << std::setw(35) << "Interleaved (stride)" << std::setw(15) << avg_ms_stride << std::setw(18) << bw_stride << std::endl;
    std::cout << std::left << std::setw(35) << "WarpID" << std::setw(15) << avg_ms_warp << std::setw(18) << bw_warp << std::endl;
    std::cout << std::left << std::setw(35) << "Standard" << std::setw(15) << avg_ms_std << std::setw(18) << bw_std << std::endl;
    std::cout << std::left << std::setw(35) << "cuBLAS" << std::setw(15) << avg_ms_cublas << std::setw(18) << bw_cublas << std::endl;
    std::cout << "========================================================" << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    cublasDestroy(cublas_handle);

    return 0;
}
