#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

/**
 * 真正工业级的 Persistent GEMV Kernel (A: MxK, x: Kx1, y: Mx1)
 * 前提假设：K 是 4 的倍数（为了 128-bit float4 向量化访存对齐）
 */
__global__ void persistentGemvCorrect(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    int* row_counter
) {
    // 申请共享内存用于两件事：
    // 1. 广播抢占到的行号
    // 2. Block 级别的归约 (存放每个 Warp 的部分和)
    __shared__ int shared_row_base;
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // 为了降低 atomicAdd 频率，每次 Block 申请连续的 4 行
    const int BATCH_SIZE = 4;

    while (true) {
        // ==========================================
        // 1. 任务抢占 (彻底杜绝 Atomic Bomb)
        // ==========================================
        // 只有 Thread 0 出面去抢任务！
        if (tid == 0) {
            shared_row_base = atomicAdd(row_counter, BATCH_SIZE);
        }
        
        // 必须同步！保证剩下的 255 个线程都能看到抢到的行号
        __syncthreads();
        
        int base_row = shared_row_base;

        // 如果抢到的起始行已经超过了矩阵的 M，说明所有任务处理完毕，全员安全撤退
        if (base_row >= M) {
            break;
        }

        // ==========================================
        // 2. 协同计算 (一个 Block 共同处理 1 行)
        // ==========================================
        for (int i = 0; i < BATCH_SIZE; i++) {
            int current_row = base_row + i;
            
            // 边界保护
            if (current_row >= M) break;

            float local_sum = 0.0f;
            const float* row_ptr = A + current_row * K;

            // 🔥 性能核心：使用 float4 进行 128-bit 向量化访存
            // 每个线程每次处理 4 个 float
            const float4* row_ptr_f4 = reinterpret_cast<const float4*>(row_ptr);
            const float4* x_f4 = reinterpret_cast<const float4*>(x);
            int K_f4 = K / 4;

            // Grid-Stride Loop: Block 内的 256 个线程平分 K 维度的计算
            for (int k = tid; k < K_f4; k += blockDim.x) {
                float4 a_vec = row_ptr_f4[k];
                float4 x_vec = x_f4[k];
                
                local_sum += a_vec.x * x_vec.x;
                local_sum += a_vec.y * x_vec.y;
                local_sum += a_vec.z * x_vec.z;
                local_sum += a_vec.w * x_vec.w;
            }

            // ==========================================
            // 3. 安全的 Block-Level 归约 (Reduction)
            // ==========================================
            // 步骤 A: Warp 内部归约 (32 归 1)
            unsigned int mask = 0xffffffff;
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                local_sum += __shfl_down_sync(mask, local_sum, offset);
            }

            // 每个 Warp 的 0 号线程把结果写到共享内存
            if (lane_id == 0) {
                warp_sums[warp_id] = local_sum;
            }
            
            // 等待所有 Warp 写完共享内存
            __syncthreads();

            // 步骤 B: Warp 间归约 (8 归 1)
            // 交给第 0 个 Warp 来完成最后的合并
            if (warp_id == 0) {
                // 读取各个 Warp 的结果，如果越界则填 0
                float final_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;
                
                #pragma unroll
                for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
                    final_sum += __shfl_down_sync(mask, final_sum, offset);
                }

                // ==========================================
                // 4. 写回结果
                // ==========================================
                // 仅由最核心的 1 个线程将最终正确的点积结果写回全局内存
                if (lane_id == 0) {
                    y[current_row] = final_sum;
                }
            }
            
            // 必须再次同步，防止进入下一行的计算时共享内存被污染
            __syncthreads(); 
        }
    }
}

int main() {
    // 1. 获取 GPU 设备信息
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    int total_sms = prop.multiProcessorCount;

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Total SMs: " << total_sms << std::endl;
    std::cout << "========================================================" << std::endl;

    // 2. 矩阵大小 (确保 K 是 4 的倍数，以便 float4 对齐)
    const int M = 8192;
    const int K = 8192;

    size_t size_A = M * K * sizeof(float);
    size_t size_x = K * sizeof(float);
    size_t size_y = M * sizeof(float);
    double total_bytes = (double)(size_A + size_x + size_y);

    std::cout << "Matrix: A[" << M << "x" << K << "], x[" << K << "], y[" << M << "]" << std::endl;
    std::cout << "========================================================" << std::endl;

    float *d_A, *d_x, *d_y;
    int *d_counter;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_x, size_x));
    CHECK_CUDA(cudaMalloc(&d_y, size_y));
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

    // 初始化数据
    CHECK_CUDA(cudaMemset(d_A, 0, size_A));
    CHECK_CUDA(cudaMemset(d_x, 0, size_x));
    CHECK_CUDA(cudaMemset(d_y, 0, size_y));

    // 3. 测试不同 SM 数量
    std::vector<int> sm_configs = {total_sms, total_sms/2, total_sms/4, total_sms/8, 16, 8, 4, 2, 1};

    // 去重排序
    std::sort(sm_configs.begin(), sm_configs.end(), std::greater<int>());
    sm_configs.erase(std::unique(sm_configs.begin(), sm_configs.end()), sm_configs.end());

    std::cout << std::left << std::setw(10) << "SM Count"
              << std::setw(15) << "Avg Time (ms)"
              << std::setw(20) << "Bandwidth (GB/s)" << std::endl;
    std::cout << "========================================================" << std::endl;

    for (int sm_count : sm_configs) {
        std::cout << "\n[SM Count: " << sm_count << "]" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        // Warmup
        const int num_warmup = 5;
        for (int i = 0; i < num_warmup; i++) {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
            persistentGemvCorrect<<<sm_count, THREADS_PER_BLOCK>>>(
                d_A, d_x, d_y, M, K, d_counter
            );
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        std::cout << "Warmup completed (" << num_warmup << " iterations)" << std::endl;

        // 正式测试 - 每次打印时间
        const int num_iters = 20;
        std::vector<float> times(num_iters);

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        for (int i = 0; i < num_iters; i++) {
            CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

            CHECK_CUDA(cudaEventRecord(start));
            persistentGemvCorrect<<<sm_count, THREADS_PER_BLOCK>>>(
                d_A, d_x, d_y, M, K, d_counter
            );
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            times[i] = ms;

            double bandwidth = (total_bytes / 1e9) / (ms / 1000.0);
            std::cout << "  Iter " << std::setw(2) << i << ": "
                      << std::fixed << std::setprecision(3) << ms << " ms, "
                      << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
        }

        // 统计
        float avg_time = 0.0f;
        for (int i = 0; i < num_iters; i++) {
            avg_time += times[i];
        }
        avg_time /= num_iters;

        float min_time = *std::min_element(times.begin(), times.end());
        float max_time = *std::max_element(times.begin(), times.end());

        double avg_bandwidth = (total_bytes / 1e9) / (avg_time / 1000.0);

        std::cout << "\n[Summary] Avg: " << std::fixed << std::setprecision(3) << avg_time << " ms"
                  << " | Min: " << min_time << " ms"
                  << " | Max: " << max_time << " ms"
                  << " | BW: " << std::fixed << std::setprecision(2) << avg_bandwidth << " GB/s"
                  << std::endl;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "Test completed!" << std::endl;

    // 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_counter));

    return 0;
}