#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 读取 GPU 时钟周期
__device__ __forceinline__ unsigned long long gpu_clock() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock));
    return clock;
}

// 获取 SM ID
__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// ---------------------------------------------------------
// Pointer Chasing Kernel - 测量 L2 延迟
// 每个 SM 独立运行，通过链表遍历测量内存访问延迟
// ---------------------------------------------------------
__global__ void l2_latency_kernel(unsigned int* next_ptr,  // 链表下一个元素的索引
                                  unsigned int* data,       // 数据数组
                                  int* latency_results,    // 结果存储
                                  int chain_length,         // 链表长度
                                  int num_sms) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    // 每个 SM 选择一个起始点
    int start_idx = my_smid * 1024;  // 每个 SM 间隔 1024 个元素
    int idx = start_idx;

    // 预热，确保数据在 L2 中
    for (int i = 0; i < 10; i++) {
        idx = next_ptr[idx];
    }

    // 开始测量
    unsigned long long start = gpu_clock();

    // Pointer chasing - 这是延迟敏感的操作
    for (int i = 0; i < chain_length; i++) {
        idx = next_ptr[idx];
    }

    unsigned long long end = gpu_clock();

    // 记录结果（每个 SM 的第一个线程）
    if (threadIdx.x == 0) {
        latency_results[my_smid] = (int)(end - start);
    }
}

// ---------------------------------------------------------
// 简化版本 - 直接测量单次全局内存访问延迟
// ---------------------------------------------------------
__global__ void l2_latency_simple(const float* __restrict__ data,
                                  float* results,
                                  int data_size,
                                  int iterations,
                                  int num_sms) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    // 每个 SM 访问不同的数据区域
    int sm_offset = my_smid * (data_size / num_sms);
    int thread_offset = threadIdx.x % (data_size / num_sms / blockDim.x);
    int base_idx = sm_offset + thread_offset;

    float sum = 0.0f;

    // 预热
    for (int i = 0; i < 100; i++) {
        sum += data[(base_idx + i * blockDim.x) % data_size];
    }
    __syncwarp();

    // 测量
    unsigned long long start = gpu_clock();

    for (int i = 0; i < iterations; i++) {
        // 随机访问模式，避免缓存行预取
        int idx = (base_idx + i * 1024) % data_size;
        sum += data[idx];
    }

    unsigned long long end = gpu_clock();

    if (threadIdx.x == 0) {
        results[my_smid] = (float)(end - start) / iterations;
    }

    // 防止编译器优化掉
    if (sum > 1e10f) results[my_smid] = 0;
}

// ---------------------------------------------------------
// 更精确的版本 - 使用原子操作序列化访问
// ---------------------------------------------------------
__global__ void l2_latency_atomic(float* __restrict__ data,
                                  int* lock,
                                  float* results,
                                  int data_size,
                                  int iterations,
                                  int num_sms) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    int sm_offset = my_smid * (data_size / num_sms);
    int base_idx = sm_offset;

    float sum = 0.0f;

    // 等待所有 SM 准备好
    atomicAdd(lock, 1);
    while (*lock < num_sms);
    __syncthreads();

    // 测量
    unsigned long long start = gpu_clock();

    for (int i = 0; i < iterations; i++) {
        int idx = (base_idx + i * 512) % data_size;
        sum += data[idx];
        __syncwarp();
    }

    unsigned long long end = gpu_clock();

    if (threadIdx.x == 0) {
        results[my_smid] = (float)(end - start) / iterations;
    }

    if (sum > 1e10f) results[my_smid] = 0;
}

// ---------------------------------------------------------
// 带宽测试 Kernel - 用于对比
// ---------------------------------------------------------
__global__ void l2_bandwidth_kernel(const float* __restrict__ A,
                                    float* __restrict__ Y,
                                    int N, int M,
                                    int num_sms) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;

    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;
    int total_active_warps = num_sms * warps_per_block;

    for (int row = global_warpID; row < N; row += total_active_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * 1.0f;
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
// 获取 GPU 信息
// ---------------------------------------------------------
void print_gpu_info() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM 数量：" << prop.multiProcessorCount << std::endl;
    std::cout << "每个 SM 最大线程数：" << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "L2 缓存大小：" << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "时钟频率：" << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------
// 测试函数
// ---------------------------------------------------------
void test_l2_latency() {
    int num_sms = 132;  // H100 的 SM 数量
    int data_size = 32 * 1024 * 1024;  // 32MB 数据，确保在 L2 中
    size_t data_bytes = data_size * sizeof(float);

    // 准备数据
    std::vector<float> h_data(data_size, 1.0f);
    std::vector<float> h_results(num_sms, 0.0f);
    std::vector<int> h_lock(1, 0);

    float *d_data;
    float *d_results;
    int *d_lock;

    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_results, num_sms * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lock, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_lock, 0, sizeof(int)));

    int threads_per_block = 256;
    int launch_blocks = num_sms;
    int iterations = 10000;

    std::cout << "=== L2 缓存延迟测试 ===" << std::endl;
    std::cout << "数据大小：" << data_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "测试 SM 数量：" << num_sms << std::endl;
    std::cout << "每 SM 迭代次数：" << iterations << std::endl;
    std::cout << std::endl;

    // 预热
    l2_latency_simple<<<launch_blocks, threads_per_block>>>(d_data, d_results, data_size, 100, num_sms);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 正式测试
    l2_latency_simple<<<launch_blocks, threads_per_block>>>(d_data, d_results, data_size, iterations, num_sms);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 读取结果
    CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, num_sms * sizeof(float), cudaMemcpyDeviceToHost));

    // 获取 GPU 时钟频率（用于转换周期为纳秒）
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    float clock_mhz = prop.clockRate / 1000.0f;
    float ns_per_cycle = 1000.0f / clock_mhz;

    // 统计分析
    float min_latency = h_results[0];
    float max_latency = h_results[0];
    float sum_latency = 0.0f;

    std::cout << "各 SM 延迟（时钟周期，越小越好）：" << std::endl;
    std::cout << std::left << std::setw(10) << "SM ID"
              << std::setw(15) << "延迟 (cycles)"
              << std::setw(15) << "延迟 (ns)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    for (int i = 0; i < num_sms; i++) {
        float latency_cycles = h_results[i];
        float latency_ns = latency_cycles * ns_per_cycle;

        if (latency_cycles < min_latency && latency_cycles > 0) min_latency = latency_cycles;
        if (latency_cycles > max_latency) max_latency = latency_cycles;
        sum_latency += latency_cycles;

        if (i < 10 || i >= num_sms - 5) {
            std::cout << std::left << std::setw(10) << i
                      << std::setw(15) << (int)latency_cycles
                      << std::setw(15) << std::fixed << std::setprecision(2) << latency_ns << std::endl;
        }
        if (i == 10) {
            std::cout << "... (省略中间部分) ..." << std::endl;
        }
    }

    float avg_latency = sum_latency / num_sms;
    std::cout << std::endl;
    std::cout << "统计结果：" << std::endl;
    std::cout << "  最小延迟：" << (int)min_latency << " cycles (" << std::fixed << std::setprecision(2) << min_latency * ns_per_cycle << " ns)" << std::endl;
    std::cout << "  最大延迟：" << (int)max_latency << " cycles (" << std::fixed << std::setprecision(2) << max_latency * ns_per_cycle << " ns)" << std::endl;
    std::cout << "  平均延迟：" << (int)avg_latency << " cycles (" << std::fixed << std::setprecision(2) << avg_latency * ns_per_cycle << " ns)" << std::endl;
    std::cout << "  延迟变化：" << std::fixed << std::setprecision(2) << (max_latency - min_latency) / avg_latency * 100 << "%" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_lock));
}

// ---------------------------------------------------------
// 带宽对比测试
// ---------------------------------------------------------
void test_l2_bandwidth() {
    int N = 4096;
    int M = 8192;
    size_t size_A = N * M * sizeof(float);
    size_t size_Y = N * sizeof(float);

    std::vector<float> h_A(N * M, 1.0f);
    std::vector<float> h_Y(N, 0.0f);

    float *d_A, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int num_sms = 132;

    double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;

    // 测试不同 SM 数量
    std::vector<int> sm_counts = {16, 33, 66, 108, 132};

    std::cout << "\n=== L2 带宽对比测试 ===" << std::endl;
    std::cout << std::left << std::setw(15) << "SM 数量"
              << std::setw(20) << "耗时 (ms)"
              << std::setw(20) << "带宽 (GB/s)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    for (int sms : sm_counts) {
        l2_bandwidth_kernel<<<num_sms, threads_per_block>>>(d_A, d_Y, N, M, sms);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            l2_bandwidth_kernel<<<num_sms, threads_per_block>>>(d_A, d_Y, N, M, sms);
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        double bw = (bytes / (ms / 1000.0)) / 1e9;

        std::cout << std::left << std::setw(15) << sms
                  << std::setw(20) << std::fixed << std::setprecision(4) << ms / iterations
                  << std::setw(20) << std::fixed << std::setprecision(2) << bw << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_Y));
}

int main() {
    print_gpu_info();

    test_l2_latency();
    test_l2_bandwidth();

    return 0;
}
