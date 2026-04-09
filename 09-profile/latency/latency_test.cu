#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>

// --- 新增的头文件 ---
#include <cstdint>  // 解决 uint32_t 未定义的问题
#include <cstdlib>  // 解决 rand() 未定义的问题
#include <cstdio>   // 解决 printf() 未定义的问题
// ------------------

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// ... 下面的 pointer_chasing_kernel 保持不变 ...

// 核心 Kernel：纯粹的指针追逐
__global__ void pointer_chasing_kernel(uint32_t *data, long long *out_time, uint32_t *out_ptr, int iters) {
    // 强制单线程
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t p = 0;

    // Warmup: 将指令加载到 I-Cache
    for (int i = 0; i < 1000; i++) {
        p = data[p];
    }

    // 记录开始周期 (clock64 防止溢出)
    long long start = clock64();

    // 循环展开，减少 for 循环本身的汇编开销
    #pragma unroll 16
    for (int i = 0; i < iters; i++) {
        p = data[p];
    }

    long long end = clock64();

    out_time[0] = end - start;
    out_ptr[0] = p; // 防止 NVCC 执行死代码消除 (Dead Code Elimination)
}

// 使用 Sattolo 算法生成单个连续的随机全局环，完美击穿预取器
void generate_single_cycle_chain(std::vector<uint32_t>& arr) {
    size_t num_elements = arr.size();
    for (size_t i = 0; i < num_elements; i++) {
        arr[i] = i;
    }
    // Sattolo's algorithm
    for (size_t i = num_elements - 1; i > 0; i--) {
        size_t j = rand() % i; 
        std::swap(arr[i], arr[j]);
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    // 记录设备时钟频率 (kHz 转为 Hz)
    double freq_hz = prop.clockRate * 1000.0; 
    
    std::cout << "Device: " << prop.name << " | Clock Rate: " << prop.clockRate / 1000.0 << " MHz\n";

    std::ofstream out_csv("latency_results.csv");
    out_csv << "Size_KB,PreferL1_ns,PreferEqual_ns,PreferShared_ns\n";

    // int iters = 100000; // 追逐次数
    // 从 2KB 测到 1048576KB (1GB)
    for (size_t size_bytes = 2048; size_bytes <= 1024ULL * 1024 * 1024; size_bytes *= 2) {
        size_t num_elements = size_bytes / sizeof(uint32_t);
            int iters = std::max(100000, (int)num_elements);

        std::vector<uint32_t> h_data(num_elements);
        generate_single_cycle_chain(h_data);

        uint32_t *d_data, *d_out_ptr;
        long long *d_out_time;
        CHECK_CUDA(cudaMalloc(&d_data, size_bytes));
        CHECK_CUDA(cudaMalloc(&d_out_time, sizeof(long long)));
        CHECK_CUDA(cudaMalloc(&d_out_ptr, sizeof(uint32_t)));

        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), size_bytes, cudaMemcpyHostToDevice));

        long long h_out_time;
        double latencies[3];

        // 依次测试三种 Cache Config
        cudaFuncCache cache_configs[3] = {cudaFuncCachePreferL1, cudaFuncCachePreferEqual, cudaFuncCachePreferShared};
        
        for (int c = 0; c < 3; c++) {
            CHECK_CUDA(cudaFuncSetCacheConfig(pointer_chasing_kernel, cache_configs[c]));
            
            // 跑两次，第一次预热 TLB
            pointer_chasing_kernel<<<1, 1>>>(d_data, d_out_time, d_out_ptr, iters);
            pointer_chasing_kernel<<<1, 1>>>(d_data, d_out_time, d_out_ptr, iters);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(&h_out_time, d_out_time, sizeof(long long), cudaMemcpyDeviceToHost));
            
            // 计算纳秒 (ns): (总周期 / 迭代次数) / 频率 * 1e9
            double ns_per_load = ((double)h_out_time / iters) / freq_hz * 1e9;
            latencies[c] = ns_per_load;
        }

        double size_kb = size_bytes / 1024.0;
        printf("Size: %8.0f KB | L1: %6.2f ns | Eq: %6.2f ns | Sh: %6.2f ns\n", 
                size_kb, latencies[0], latencies[1], latencies[2]);
                
        out_csv << size_kb << "," << latencies[0] << "," << latencies[1] << "," << latencies[2] << "\n";

        CHECK_CUDA(cudaFree(d_data));
        CHECK_CUDA(cudaFree(d_out_time));
        CHECK_CUDA(cudaFree(d_out_ptr));
    }

    out_csv.close();
    return 0;
}