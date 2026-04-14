/**
 * Persistent Kernel for GEMV - 验证 SM 数量对性能的影响
 *
 * Persistent kernel 特点:
 * 1. 启动固定数量的线程块，每个块对应一个 SM
 * 2. 每个块持续运行，轮流处理多个 GEMV 任务
 * 3. 通过 gridDim.x 精确控制使用的 SM 数量
 * 4. 避免传统 kernel 的 launch overhead 和调度开销
 *
 * GEMV: y = A * x, 其中 A 是 MxK 矩阵，x 是 Kx1 向量，y 是 Mx1 向量
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 配置参数
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ROWS_PER_WARP = 4;  // 每个 warp 处理的行数

/**
 * Persistent GEMV Kernel
 *
 * 设计思路:
 * - 每个线程块负责处理一部分输出行
 * - 使用 persistent 模式，块持续运行直到所有行处理完毕
 * - 通过 atomic 计数器动态分配工作，实现负载均衡
 *
 * @param A 输入矩阵 [M, K], 行主序
 * @param x 输入向量 [K]
 * @param y 输出向量 [M]
 * @param M 矩阵 A 的行数
 * @param K 矩阵 A 的列数
 * @param row_counter 原子计数器，用于动态分配行
 */
__global__ void persistentGemvKernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    int* row_counter
) {
    // 每个块处理的行数 = gridDim.x 个块平分 M 行
    // 使用动态调度：每次从计数器中获取一批行来处理

    const int batch_size = 32;  // 每次获取的行数，用于负载均衡
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // 共享内存用于缓存 x 向量的部分数据（可选优化）
    extern __shared__ float shared_x[];

    // 加载 x 到共享内存（如果 K 较小，可以全部加载）
    // 这里采用直接读取全局内存的简单策略

    while (true) {
        // 原子获取下一批行
        int row_start = atomicAdd(row_counter, batch_size);
        if (row_start >= M) {
            break;  // 所有行已处理完毕
        }

        int row_end = min(row_start + batch_size, M);

        // 当前块负责的行索引
        int my_row = row_start + block_id;

        // 遍历分配给当前块的行
        while (my_row < row_end) {
            float sum = 0.0f;

            // 计算点积：A[my_row, :] · x
            const float* row_ptr = A + my_row * K;

            // 简单的串行点积（可以进一步优化为并行归约）
            for (int k = 0; k < K; k++) {
                sum += row_ptr[k] * x[k];
            }

            y[my_row] = sum;

            my_row += num_blocks;
        }

        // 确保所有块完成当前批次
        __syncthreads();
    }
}

/**
 * 优化的 Persistent GEMV Kernel - 使用 Warp 级别的并行
 *
 * 每个 warp 负责多行的计算，线程在 warp 内协作
 */
__global__ void persistentGemvOptimized(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    int* row_counter
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    const int batch_size = 64;

    int num_blocks = gridDim.x;

    while (true) {
        int row_start = atomicAdd(row_counter, batch_size);
        if (row_start >= M) {
            break;
        }

        int row_end = min(row_start + batch_size, M);

        // 当前块的起始行 + warp 分配
        int base_row = row_start + blockIdx.x;

        while (base_row < row_end) {
            // 每个 warp 处理 ROWS_PER_WARP 行
            for (int warp_row = 0; warp_row < ROWS_PER_WARP; warp_row++) {
                int my_row = base_row + warp_id * ROWS_PER_WARP + warp_row;
                if (my_row >= row_end || my_row >= M) {
                    // 考虑跨块的情况
                    my_row += num_blocks * warps_per_block * ROWS_PER_WARP;
                    if (my_row >= row_end && my_row >= M) {
                        continue;
                    }
                }

                if (my_row < M) {
                    float sum = 0.0f;
                    const float* row_ptr = A + my_row * K;

                    // 使用 warp 内线程并行计算点积
                    for (int k = lane_id; k < K; k += WARP_SIZE) {
                        sum += row_ptr[k] * x[k];
                    }

                    // Warp 内归约
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                        sum += __shfl_down_sync(0xffffffff, sum, offset);
                    }

                    // lane 0 写入结果
                    if (lane_id == 0) {
                        y[my_row] = sum;
                    }
                }
            }

            base_row += num_blocks * warps_per_block * ROWS_PER_WARP;
        }
    }
}

/**
 * 数据初始化 kernel
 */
__global__ void initData(float* data, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

/**
 * 验证结果正确性
 */
bool verifyResult(const float* y_gpu, const float* y_cpu, int M, float tolerance = 1e-3) {
    std::vector<float> h_y(M);
    CHECK_CUDA(cudaMemcpy(h_y.data(), y_gpu, M * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++) {
        float diff = std::abs(h_y[i] - y_cpu[i]);
        if (diff > tolerance) {
            std::cerr << "Verification failed at index " << i
                      << ": GPU=" << h_y[i] << ", CPU=" << y_cpu[i]
                      << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * CPU 参考实现
 */
void gemvCpu(const float* A, const float* x, float* y, int M, int K) {
    for (int m = 0; m < M; m++) {
        y[m] = 0.0f;
        for (int k = 0; k < K; k++) {
            y[m] += A[m * K + k] * x[k];
        }
    }
}

int main() {
    // 1. 获取 GPU 设备信息
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    int total_sms = prop.multiProcessorCount;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Total SMs: " << total_sms << std::endl;
    std::cout << "Max threads/SM: " << max_threads_per_sm << std::endl;
    std::cout << "Max blocks/SM: " << max_blocks_per_sm << std::endl;
    std::cout << "========================================================" << std::endl;

    // 2. 定义矩阵维度
    const int M = 8192;   // 输出行数
    const int K = 8192;   // 输入维度

    size_t size_A = M * K * sizeof(float);
    size_t size_x = K * sizeof(float);
    size_t size_y = M * sizeof(float);
    double total_bytes = (double)(size_A + size_x + size_y);

    std::cout << "\nMatrix Size: A[" << M << " x " << K << "], x[" << K << "], y[" << M << "]" << std::endl;
    std::cout << "Total Memory: " << (total_bytes / 1e6) << " MB" << std::endl;
    std::cout << "========================================================" << std::endl;

    // 3. 分配显存
    float *d_A, *d_x, *d_y;
    int *d_counter;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_x, size_x));
    CHECK_CUDA(cudaMalloc(&d_y, size_y));
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

    // 4. 初始化数据
    int threads = 256;
    initData<<<(M * K + threads - 1) / threads, threads>>>(d_A, M * K, 0.5f);
    initData<<<(K + threads - 1) / threads, threads>>>(d_x, K, 2.0f);
    initData<<<(M + threads - 1) / threads, threads>>>(d_y, M, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. CPU 参考计算（用于验证）
    std::vector<float> h_A(M * K, 0.5f);
    std::vector<float> h_x(K, 2.0f);
    std::vector<float> y_cpu(M);
    gemvCpu(h_A.data(), h_x.data(), y_cpu.data(), M, K);

    // 6. 验证 baseline (使用所有 SM) 的正确性
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    size_t shared_mem = 0;  // 简单版本不使用共享内存

    persistentGemvKernel<<<total_sms, THREADS_PER_BLOCK, shared_mem>>>(
        d_A, d_x, d_y, M, K, d_counter
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verifyResult(d_y, y_cpu.data(), M)) {
        std::cout << "\n✓ Kernel verification passed!" << std::endl;
    } else {
        std::cout << "\n✗ Kernel verification failed!" << std::endl;
        return 1;
    }

    // 7. 测试不同 SM 数量的性能
    std::vector<int> sm_configs;

    // 生成测试配置：从 1 到 total_sms
    sm_configs.push_back(total_sms);  // 全部 SM
    sm_configs.push_back(total_sms / 2);
    sm_configs.push_back(total_sms / 4);
    sm_configs.push_back(total_sms / 8);
    sm_configs.push_back(16);
    sm_configs.push_back(8);
    sm_configs.push_back(4);
    sm_configs.push_back(2);
    sm_configs.push_back(1);

    // 移除重复和无效值
    std::sort(sm_configs.begin(), sm_configs.end(), std::greater<int>());
    sm_configs.erase(std::unique(sm_configs.begin(), sm_configs.end()), sm_configs.end());

    std::cout << "\n========================================================" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << std::left
              << std::setw(12) << "SM Count"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Bandwidth (GB/s)"
              << std::setw(15) << "Efficiency"
              << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    float baseline_time = 0.0f;

    for (int sm_count : sm_configs) {
        // 创建 NVTX range 便于分析
        std::string rangeName = "Persistent_SM_" + std::to_string(sm_count);
        nvtxRangePushA(rangeName.c_str());

        // 重置 counter 和输出
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_y, 0, size_y));

        // 创建事件用于计时
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // 预热
        for (int i = 0; i < 3; i++) {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            persistentGemvKernel<<<sm_count, THREADS_PER_BLOCK, 0>>>(
                d_A, d_x, d_y, M, K, d_counter
            );
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // 正式计时
        const int num_iters = 20;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < num_iters; i++) {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            persistentGemvKernel<<<sm_count, THREADS_PER_BLOCK, 0>>>(
                d_A, d_x, d_y, M, K, d_counter
            );
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / num_iters;

        // 计算带宽
        double bandwidth = (total_bytes / 1e9) / (avg_ms / 1000.0);

        // 计算效率（相对于 baseline）
        float efficiency = 100.0f;
        if (baseline_time > 0) {
            efficiency = (baseline_time / avg_ms) * 100.0f;
        } else {
            baseline_time = avg_ms;
        }

        std::cout << std::left
                  << std::setw(12) << sm_count
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_ms
                  << std::setw(20) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(14) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << std::endl;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        nvtxRangePop();
    }

    std::cout << "========================================================" << std::endl;

    // 8. 额外测试：不同问题规模下的表现
    std::cout << "\nScaling Behavior (Different Problem Sizes):" << std::endl;
    std::cout << "========================================================" << std::endl;

    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    std::cout << std::left
              << std::setw(12) << "Size (M=K)"
              << std::setw(15) << "1 SM (ms)"
              << std::setw(15) << "All SMs (ms)"
              << std::setw(15) << "Speedup"
              << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int size : sizes) {
        size_t test_A_size = size * size * sizeof(float);
        size_t test_x_size = size * sizeof(float);
        size_t test_y_size = size * sizeof(float);
        double test_bytes = (double)(test_A_size + test_x_size + test_y_size);

        float *d_A_test, *d_x_test, *d_y_test;
        CHECK_CUDA(cudaMalloc(&d_A_test, test_A_size));
        CHECK_CUDA(cudaMalloc(&d_x_test, test_x_size));
        CHECK_CUDA(cudaMalloc(&d_y_test, test_y_size));

        initData<<<(size * size + threads - 1) / threads, threads>>>(d_A_test, size * size, 0.5f);
        initData<<<(size + threads - 1) / threads, threads>>>(d_x_test, size, 2.0f);

        // 1 SM
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        persistentGemvKernel<<<1, THREADS_PER_BLOCK, 0>>>(
            d_A_test, d_x_test, d_y_test, size, size, d_counter
        );

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            persistentGemvKernel<<<1, THREADS_PER_BLOCK, 0>>>(
                d_A_test, d_x_test, d_y_test, size, size, d_counter
            );
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time_1sm = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_1sm, start, stop));
        time_1sm /= 10;

        // All SMs
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            persistentGemvKernel<<<total_sms, THREADS_PER_BLOCK, 0>>>(
                d_A_test, d_x_test, d_y_test, size, size, d_counter
            );
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time_all = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_all, start, stop));
        time_all /= 10;

        float speedup = time_1sm / time_all;

        std::cout << std::left
                  << std::setw(12) << size
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_1sm
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_all
                  << std::setw(14) << std::fixed << std::setprecision(1) << speedup << "x"
                  << std::endl;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A_test));
        CHECK_CUDA(cudaFree(d_x_test));
        CHECK_CUDA(cudaFree(d_y_test));
    }

    std::cout << "========================================================" << std::endl;

    // 9. 清理资源
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_counter));

    std::cout << "\nTest completed successfully!" << std::endl;

    return 0;
}
