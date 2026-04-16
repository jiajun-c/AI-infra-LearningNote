#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstring>

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

__device__ __forceinline__ unsigned long long gpu_clock() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock));
    return clock;
}

// ---------------------------------------------------------
// Pointer Chasing - 精确测量延迟
// 每个链表在一个 L2 slice 内，测试不同 SM 访问不同 slice 的延迟
// ---------------------------------------------------------
__global__ void pointer_chasing_kernel(unsigned int** next_ptr,
                                       int* results,
                                       int chain_length,
                                       int num_sms,
                                       int num_slices) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    // 每个 SM 测试访问不同 slice 的延迟
    for (int slice = 0; slice < num_slices; slice++) {
        // 计算要访问的链表
        unsigned int** list = &next_ptr[slice * num_sms + my_smid];
        unsigned int* chain = *list;

        // 预热
        unsigned int idx = 0;
        for (int i = 0; i < 10; i++) {
            idx = chain[idx];
        }

        // 测量
        unsigned long long start = gpu_clock();
        for (int i = 0; i < chain_length; i++) {
            idx = chain[idx];
        }
        unsigned long long end = gpu_clock();

        if (threadIdx.x == 0) {
            results[my_smid * num_slices + slice] = (int)(end - start);
        }
    }
}

// ---------------------------------------------------------
// 简化版：固定地址跨 slice 访问
// ---------------------------------------------------------
__global__ void cross_slice_latency_kernel(const float* __restrict__ data,
                                           int* results,
                                           int data_size,
                                           int iterations,
                                           int num_sms,
                                           int num_slices,
                                           int slice_size) {
    unsigned int my_smid = get_smid();

    if (my_smid >= num_sms) return;

    // 每个 SM 访问所有 slice，测量延迟
    for (int slice = 0; slice < num_slices; slice++) {
        int base_offset = slice * slice_size + (my_smid % slice_size);
        float sum = 0.0f;

        // 预热
        for (int i = 0; i < 50; i++) {
            sum += data[base_offset + (i * 128) % slice_size];
        }
        __syncwarp();

        // 测量
        unsigned long long start = gpu_clock();
        for (int i = 0; i < iterations; i++) {
            sum += data[base_offset + (i * 128) % slice_size];
        }
        unsigned long long end = gpu_clock();

        if (threadIdx.x == 0) {
            results[my_smid * num_slices + slice] = (int)(end - start);
        }

        if (sum > 1e10f) results[0] = 0;
    }
}

// ---------------------------------------------------------
// SM 分组测试 - 测试不同 SM 组访问同一 L2 slice
// ---------------------------------------------------------
__global__ void sm_group_kernel(const float* __restrict__ data,
                                int* results,
                                int data_size,
                                int iterations,
                                int target_slice,
                                int num_slices,
                                int slice_size) {
    unsigned int my_smid = get_smid();

    // 只让部分 SM 工作
    int sms_per_slice = 3;  // 假设每个 slice 对应 3 个 SM
    int base_sm = target_slice * sms_per_slice;

    if (my_smid < base_sm || my_smid >= base_sm + sms_per_slice) return;

    int sm_offset_in_slice = my_smid - base_sm;
    int base_offset = target_slice * slice_size + sm_offset_in_slice * 1024;

    float sum = 0.0f;

    // 预热
    for (int i = 0; i < 50; i++) {
        sum += data[base_offset + (i * 64) % slice_size];
    }
    __syncwarp();

    // 测量
    unsigned long long start = gpu_clock();
    for (int i = 0; i < iterations; i++) {
        sum += data[base_offset + (i * 64) % slice_size];
    }
    unsigned long long end = gpu_clock();

    if (threadIdx.x == 0) {
        results[my_smid] = (int)(end - start);
    }

    if (sum > 1e10f) results[0] = 0;
}

void print_gpu_info() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM 数量：" << prop.multiProcessorCount << std::endl;
    std::cout << "L2 缓存大小：" << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "时钟频率：" << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------
// 测试：每个 SM 访问不同 L2 slice 的延迟矩阵
// ---------------------------------------------------------
void test_cross_slice_latency() {
    int num_sms = 132;
    int num_slices = 48;  // H100 的 L2 slice 数量
    int slice_size = 1024 * 1024;  // 每个 slice 1M floats = 4MB
    int data_size = num_slices * slice_size;
    size_t data_bytes = data_size * sizeof(float);

    std::cout << "=== 跨 Slice 延迟矩阵测试 ===" << std::endl;
    std::cout << "数据大小：" << data_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "L2 Slice 数量：" << num_slices << std::endl;
    std::cout << "每 Slice 大小：" << slice_size * 4 / 1024 << " MB" << std::endl;
    std::cout << std::endl;

    std::vector<float> h_data(data_size, 1.0f);
    std::vector<int> h_results(num_sms * num_slices, 0);

    float *d_data;
    int *d_results;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_results, num_sms * num_slices * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int iterations = 2000;

    // 预热
    cross_slice_latency_kernel<<<num_sms, threads_per_block>>>(d_data, d_results, data_size, 100, num_sms, num_slices, slice_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 正式测试
    cross_slice_latency_kernel<<<num_sms, threads_per_block>>>(d_data, d_results, data_size, iterations, num_sms, num_slices, slice_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, num_sms * num_slices * sizeof(int), cudaMemcpyDeviceToHost));

    // 获取时钟频率
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    float clock_mhz = prop.clockRate / 1000.0f;
    float ns_per_cycle = 1000.0f / clock_mhz;

    // 分析：每个 SM 访问不同 slice 的延迟
    std::cout << "前 8 个 SM 访问各 Slice 的延迟（cycles）：" << std::endl;
    std::cout << std::left << std::setw(8) << "SM";

    for (int s = 0; s < 8; s++) {
        std::cout << std::setw(8) << ("S" + std::to_string(s));
    }
    std::cout << std::setw(12) << "平均" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int sm = 0; sm < 8; sm++) {
        int sum = 0;
        int min_val = h_results[sm * num_slices];
        int max_val = h_results[sm * num_slices];
        int best_slice = 0;
        int worst_slice = 0;

        for (int slice = 0; slice < num_slices; slice++) {
            int val = h_results[sm * num_slices + slice];
            sum += val;
            if (val < min_val) { min_val = val; best_slice = slice; }
            if (val > max_val) { max_val = val; worst_slice = slice; }
        }
        float avg = (float)sum / num_slices;

        std::cout << std::left << std::setw(8) << sm;
        for (int slice = 0; slice < 8; slice++) {
            std::cout << std::setw(8) << h_results[sm * num_slices + slice];
        }
        std::cout << std::setw(12) << (int)avg;
        std::cout << " [min=" << min_val << "(S" << best_slice << "), max=" << max_val << "(S" << worst_slice << ")]" << std::endl;
    }

    // 统计分析 locality 效应
    float total_best = 0, total_worst = 0;
    for (int sm = 0; sm < num_sms; sm++) {
        int min_val = h_results[sm * num_slices];
        int max_val = h_results[sm * num_slices];
        for (int slice = 0; slice < num_slices; slice++) {
            if (h_results[sm * num_slices + slice] < min_val) min_val = h_results[sm * num_slices + slice];
            if (h_results[sm * num_slices + slice] > max_val) max_val = h_results[sm * num_slices + slice];
        }
        total_best += min_val;
        total_worst += max_val;
    }
    float avg_best = total_best / num_sms;
    float avg_worst = total_worst / num_sms;
    float locality_pct = (avg_worst - avg_best) / avg_best * 100;

    std::cout << std::endl;
    std::cout << "=== 统计结果 ===" << std::endl;
    std::cout << "最佳 slice 平均延迟：" << (int)avg_best << " cycles (" << avg_best * ns_per_cycle << " ns)" << std::endl;
    std::cout << "最差 slice 平均延迟：" << (int)avg_worst << " cycles (" << avg_worst * ns_per_cycle << " ns)" << std::endl;
    std::cout << "Locality 效应：" << std::fixed << std::setprecision(2) << locality_pct << "%" << std::endl;

    if (locality_pct > 5) {
        std::cout << "结论：检测到明显的 L2 Locality 效应！" << std::endl;
    } else if (locality_pct > 1) {
        std::cout << "结论：L2 Locality 效应较弱（" << std::fixed << std::setprecision(1) << locality_pct << "%）" << std::endl;
    } else {
        std::cout << "结论：未检测到明显 L2 Locality 效应" << std::endl;
        std::cout << "       Hopper 的 L2 可能采用全互联或均匀分布设计" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_results));
}

// ---------------------------------------------------------
// 测试：SM 分组访问同一 slice
// ---------------------------------------------------------
void test_sm_group_locality() {
    int num_slices = 48;
    int slice_size = 2 * 1024 * 1024;  // 2MB per slice
    int data_size = num_slices * slice_size;
    size_t data_bytes = data_size * sizeof(float);

    std::cout << "\n=== SM 分组 Locality 测试 ===" << std::endl;
    std::cout << "测试每个 L2 slice 对应的 3 个 SM 的访问延迟" << std::endl;
    std::cout << std::endl;

    std::vector<float> h_data(data_size, 1.0f);
    std::vector<int> h_results(132, 0);

    float *d_data;
    int *d_results;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_results, 132 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int iterations = 5000;

    std::cout << std::left << std::setw(12) << "Slice"
              << std::setw(12) << "SM"
              << std::setw(15) << "延迟 (c)"
              << std::setw(15) << "延迟 (ns)"
              << std::setw(12) << "相对差异" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (int slice = 0; slice < 8; slice++) {  // 测试前 8 个 slice
        sm_group_kernel<<<132, threads_per_block>>>(d_data, d_results, data_size, iterations, slice, num_slices, slice_size);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, 132 * sizeof(int), cudaMemcpyDeviceToHost));

        int base_sm = slice * 3;
        int ref_delay = h_results[base_sm];

        for (int sm = base_sm; sm < base_sm + 3; sm++) {
            int delay = h_results[sm];
            float rel_diff = (float)(delay - ref_delay) / ref_delay * 100;

            int device;
            CHECK_CUDA(cudaGetDevice(&device));
            cudaDeviceProp prop;
            CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
            float clock_mhz = prop.clockRate / 1000.0f;
            float ns_per_cycle = 1000.0f / clock_mhz;

            std::cout << std::left << std::setw(12) << slice
                      << std::setw(12) << sm
                      << std::setw(15) << delay
                      << std::setw(15) << std::fixed << std::setprecision(2) << delay * ns_per_cycle
                      << std::setw(12) << std::fixed << std::setprecision(1) << rel_diff << "%" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "结论：如果同组 SM 间延迟差异 <5%，则 L2 locality 效应不明显" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_results));
}

int main() {
    print_gpu_info();

    test_cross_slice_latency();
    test_sm_group_locality();

    return 0;
}
