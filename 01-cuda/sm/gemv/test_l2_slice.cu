#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 获取 SM ID
__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// 读取 GPU 时钟
__device__ __forceinline__ unsigned long long gpu_clock() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock));
    return clock;
}

// ---------------------------------------------------------
// L2 Slice 延迟测试 Kernel
// 通过控制访问地址，测试不同 SM 访问不同 L2 slice 的延迟
// ---------------------------------------------------------
__global__ void l2_slice_latency_kernel(const float* __restrict__ data,
                                        float* results,
                                        int data_size,
                                        int iterations,
                                        int num_sms,
                                        int l2_slices) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    // 每个 SM 有多个线程，每个线程测试不同的 L2 slice
    int threads_per_sm = blockDim.x;

    // 计算当前线程要测试的 L2 slice ID
    // 假设 L2 slice 按地址范围划分，每个 slice 负责 data_size / l2_slices 的数据
    int slice_per_thread = l2_slices / threads_per_sm;
    int base_slice = (threadIdx.x * slice_per_thread) % l2_slices;

    // 计算该 slice 对应的数据区域起始偏移
    int slice_size = data_size / l2_slices;
    int base_offset = base_slice * slice_size + threadIdx.x;

    float sum = 0.0f;

    // 预热
    for (int i = 0; i < 50; i++) {
        int idx = (base_offset + i * 32) % data_size;
        sum += data[idx];
    }
    __syncwarp();

    // 测量 - 固定访问同一个 slice 内的数据
    unsigned long long start = gpu_clock();

    for (int i = 0; i < iterations; i++) {
        // stride 访问，但保持在同一个 slice 内
        int idx = (base_offset + i * 64) % (slice_size) + base_slice * slice_size;
        sum += data[idx];
    }

    unsigned long long end = gpu_clock();

    // 每个线程记录自己测试的 slice 的延迟
    if (threadIdx.x < l2_slices && my_smid < 8) {  // 只记录前 8 个 SM 的数据
        results[my_smid * l2_slices + threadIdx.x] = (float)(end - start) / iterations;
    }

    // 防止优化
    if (sum > 1e10f) results[0] = 0;
}

// ---------------------------------------------------------
// 简化版本：测试"本地"vs"远程"L2 访问
// ---------------------------------------------------------
__global__ void l2_locality_kernel(const float* __restrict__ data,
                                   float* local_results,
                                   float* remote_results,
                                   int data_size,
                                   int iterations,
                                   int num_sms,
                                   int l2_slices) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    int slice_size = data_size / l2_slices;

    // 假设 SM 和 L2 slice 有某种映射关系
    // Hopper 上通常 SM i 访问 L2 slice i % l2_slices 最快
    int local_slice = my_smid % l2_slices;
    int remote_slice = (local_slice + l2_slices / 2) % l2_slices;

    int local_offset = local_slice * slice_size + threadIdx.x;
    int remote_offset = remote_slice * slice_size + threadIdx.x;

    float local_sum = 0.0f;
    float remote_sum = 0.0f;

    // 预热
    for (int i = 0; i < 50; i++) {
        local_sum += data[(local_offset + i * 32) % data_size];
        remote_sum += data[(remote_offset + i * 32) % data_size];
    }
    __syncwarp();

    // 测量本地访问
    unsigned long long start = gpu_clock();
    for (int i = 0; i < iterations; i++) {
        local_sum += data[(local_offset + i * 64) % (slice_size) + local_slice * slice_size];
    }
    unsigned long long end = gpu_clock();

    if (threadIdx.x == 0) {
        local_results[my_smid] = (float)(end - start) / iterations;
    }

    // 测量远程访问
    start = gpu_clock();
    for (int i = 0; i < iterations; i++) {
        remote_sum += data[(remote_offset + i * 64) % (slice_size) + remote_slice * slice_size];
    }
    end = gpu_clock();

    if (threadIdx.x == 0) {
        remote_results[my_smid] = (float)(end - start) / iterations;
    }

    if (local_sum > 1e10f || remote_sum > 1e10f) local_results[0] = 0;
}

// ---------------------------------------------------------
// L2 Slice 带宽测试 - 测试跨 slice 访问的带宽
// ---------------------------------------------------------
__global__ void l2_slice_bandwidth_kernel(const float* __restrict__ A,
                                          float* __restrict__ Y,
                                          int N, int M,
                                          int target_slice,
                                          int l2_slices,
                                          int data_per_slice) {
    unsigned int my_smid = get_smid();

    // 只让特定 slice 对应的 SM 工作
    int sm_per_slice = 132 / l2_slices;
    int slice_start_sm = target_slice * sm_per_slice;
    int slice_end_sm = slice_start_sm + sm_per_slice;

    if (my_smid < slice_start_sm || my_smid >= slice_end_sm) return;

    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    int global_warpID = (my_smid - slice_start_sm) * warps_per_block + local_warpID;
    int total_warps = sm_per_slice * warps_per_block;

    for (int row = global_warpID; row < N; row += total_warps) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            // 强制访问目标 slice 的数据
            int slice_offset = target_slice * data_per_slice;
            int col_in_slice = (col + slice_offset) % M;
            partial_sum += A[row * M + col_in_slice];
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
    std::cout << "L2 缓存大小：" << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "时钟频率：" << prop.clockRate / 1000 << " MHz" << std::endl;

    // Hopper 的 L2 slice 数量（需要查文档或实验确定）
    // H100 通常有 48 个 L2 slice
    int l2_slices = 48;
    std::cout << "L2 Slice 数量（估计）：" << l2_slices << std::endl;
    std::cout << "每 Slice 大小：" << (prop.l2CacheSize / 1024) / l2_slices << " KB" << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------
// 测试 L2 locality 效应
// ---------------------------------------------------------
void test_l2_locality() {
    int num_sms = 132;
    int l2_slices = 48;  // H100 的 L2 slice 数量
    int data_size = 48 * 1024 * 1024;  // 48MB，每个 slice 约 1MB
    size_t data_bytes = data_size * sizeof(float);

    std::vector<float> h_data(data_size, 1.0f);
    std::vector<float> h_local(num_sms, 0.0f);
    std::vector<float> h_remote(num_sms, 0.0f);

    float *d_data, *d_local, *d_remote;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_local, num_sms * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_remote, num_sms * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int launch_blocks = num_sms;
    int iterations = 5000;

    std::cout << "=== L2 Locality 效应测试 ===" << std::endl;
    std::cout << "数据大小：" << data_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "L2 Slice 数量：" << l2_slices << std::endl;
    std::cout << "每 Slice 数据：" << data_bytes / l2_slices / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << std::endl;

    // 预热
    l2_locality_kernel<<<launch_blocks, threads_per_block>>>(d_data, d_local, d_remote, data_size, 100, num_sms, l2_slices);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 正式测试
    l2_locality_kernel<<<launch_blocks, threads_per_block>>>(d_data, d_local, d_remote, data_size, iterations, num_sms, l2_slices);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_local.data(), d_local, num_sms * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_remote.data(), d_remote, num_sms * sizeof(float), cudaMemcpyDeviceToHost));

    // 获取时钟频率
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    float clock_mhz = prop.clockRate / 1000.0f;
    float ns_per_cycle = 1000.0f / clock_mhz;

    // 统计
    float local_sum = 0, remote_sum = 0;
    float local_min = h_local[0], local_max = h_local[0];
    float remote_min = h_remote[0], remote_max = h_remote[0];

    std::cout << std::left << std::setw(8) << "SM ID"
              << std::setw(15) << "本地延迟 (c)"
              << std::setw(15) << "本地延迟 (ns)"
              << std::setw(15) << "远程延迟 (c)"
              << std::setw(15) << "远程延迟 (ns)"
              << std::setw(12) << "差异 (%)" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < num_sms; i++) {
        local_sum += h_local[i];
        remote_sum += h_remote[i];

        if (h_local[i] < local_min) local_min = h_local[i];
        if (h_local[i] > local_max) local_max = h_local[i];
        if (h_remote[i] < remote_min) remote_min = h_remote[i];
        if (h_remote[i] > remote_max) remote_max = h_remote[i];

        float diff_pct = (h_remote[i] - h_local[i]) / h_local[i] * 100;

        if (i < 8 || i >= num_sms - 3) {
            std::cout << std::left << std::setw(8) << i
                      << std::setw(15) << (int)h_local[i]
                      << std::setw(15) << std::fixed << std::setprecision(2) << h_local[i] * ns_per_cycle
                      << std::setw(15) << (int)h_remote[i]
                      << std::setw(15) << std::fixed << std::setprecision(2) << h_remote[i] * ns_per_cycle
                      << std::setw(12) << std::fixed << std::setprecision(1) << diff_pct << "%" << std::endl;
        }
        if (i == 8) {
            std::cout << "... (省略中间部分) ..." << std::endl;
        }
    }

    float local_avg = local_sum / num_sms;
    float remote_avg = remote_sum / num_sms;
    float locality_effect = (remote_avg - local_avg) / local_avg * 100;

    std::cout << std::endl;
    std::cout << "=== 统计结果 ===" << std::endl;
    std::cout << "本地访问平均延迟：" << std::fixed << std::setprecision(2) << local_avg << " cycles (" << local_avg * ns_per_cycle << " ns)" << std::endl;
    std::cout << "远程访问平均延迟：" << std::fixed << std::setprecision(2) << remote_avg << " cycles (" << remote_avg * ns_per_cycle << " ns)" << std::endl;
    std::cout << "Locality 效应：" << std::fixed << std::setprecision(2) << locality_effect << "%" << std::endl;
    std::cout << std::endl;

    if (locality_effect > 5) {
        std::cout << "结论：检测到明显的 L2 Locality 效应！" << std::endl;
        std::cout << "     远程访问比本地访问慢约 " << std::fixed << std::setprecision(1) << locality_effect << "%" << std::endl;
    } else if (locality_effect > 0) {
        std::cout << "结论：L2 Locality 效应不明显（<" << std::fixed << std::setprecision(0) << locality_effect << "%）" << std::endl;
    } else {
        std::cout << "结论：未检测到 L2 Locality 效应，可能映射假设不正确" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_local));
    CHECK_CUDA(cudaFree(d_remote));
}

// ---------------------------------------------------------
// 测试不同 L2 slice 的带宽
// ---------------------------------------------------------
void test_l2_slice_bandwidth() {
    int N = 8192;
    int M = 8192;
    size_t size_A = N * M * sizeof(float);
    size_t size_Y = N * sizeof(float);
    int l2_slices = 48;
    int data_per_slice = (N * M) / l2_slices;

    std::vector<float> h_A(N * M, 1.0f);
    std::vector<float> h_Y(N, 0.0f);

    float *d_A, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int launch_blocks = 132;

    double bytes = (double)N * M * 4.0 + M * 4.0 + N * 4.0;

    std::cout << "=== L2 Slice 带宽分布测试 ===" << std::endl;
    std::cout << "矩阵大小：" << N << " x " << M << std::endl;
    std::cout << "测试 " << l2_slices << " 个 L2 slice" << std::endl;
    std::cout << std::endl;

    std::vector<float> slice_bw(l2_slices, 0);

    for (int slice = 0; slice < l2_slices; slice++) {
        l2_slice_bandwidth_kernel<<<launch_blocks, threads_per_block>>>(d_A, d_Y, N, M, slice, l2_slices, data_per_slice);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 50;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            l2_slice_bandwidth_kernel<<<launch_blocks, threads_per_block>>>(d_A, d_Y, N, M, slice, l2_slices, data_per_slice);
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        slice_bw[slice] = (bytes / (ms / 1000.0)) / 1e9;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 统计分析
    float min_bw = slice_bw[0], max_bw = slice_bw[0], sum_bw = 0;
    for (int i = 0; i < l2_slices; i++) {
        if (slice_bw[i] < min_bw) min_bw = slice_bw[i];
        if (slice_bw[i] > max_bw) max_bw = slice_bw[i];
        sum_bw += slice_bw[i];
    }
    float avg_bw = sum_bw / l2_slices;

    std::cout << "Slice 带宽统计：" << std::endl;
    std::cout << "  最小：" << std::fixed << std::setprecision(2) << min_bw << " GB/s" << std::endl;
    std::cout << "  最大：" << std::fixed << std::setprecision(2) << max_bw << " GB/s" << std::endl;
    std::cout << "  平均：" << std::fixed << std::setprecision(2) << avg_bw << " GB/s" << std::endl;
    std::cout << "  变化：" << std::fixed << std::setprecision(2) << (max_bw - min_bw) / avg_bw * 100 << "%" << std::endl;

    // 打印分布
    std::cout << std::endl;
    std::cout << "L2 Slice 带宽分布（每 8 个显示一个）：" << std::endl;
    for (int i = 0; i < l2_slices; i += 8) {
        std::cout << "Slice " << std::setw(2) << i << ": " << std::fixed << std::setprecision(2) << slice_bw[i] << " GB/s" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_Y));
}

int main() {
    print_gpu_info();

    test_l2_locality();
    std::cout << std::endl;
    test_l2_slice_bandwidth();

    return 0;
}
