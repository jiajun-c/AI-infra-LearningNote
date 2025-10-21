#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cuda/atomic>
#include <cuda/std/barrier>
// 简单的向量加法计算函数
__device__ void compute(int* global_out, const int* shared_in, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        global_out[idx] =  shared_in[idx]*2; // 简单的加1操作
    }
}

// 使用memcpy_async的内核函数
__global__ void with_memcpy_async(int* global_out, const int* global_in, size_t size, size_t batch_sz, 
                                  float* kernel_time_ms) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

    size_t local_idx = block.thread_rank();

    for (size_t batch = 0; batch < batch_sz; ++batch) {
    // Compute the index of the current batch for this block in global memory:
        size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
        size_t global_idx = block_batch_idx + threadIdx.x;
        shared[local_idx] = global_in[global_idx];

        block.sync(); // Wait for all copies to complete

        compute(global_out + block_batch_idx, shared, block.size()); // Compute and write result to global memory

        block.sync(); 
    }
}

// 传统同步版本的内核函数（用于对比）
__global__ void traditional_sync(int* global_out, const int* global_in, size_t size, size_t batch_sz,
                                float* kernel_time_ms) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

    size_t local_idx = block.thread_rank();

    for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    // Whole thread-group cooperatively copies whole batch to shared memory:
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx,cuda::aligned_size_t<16>( sizeof(int) * block.size()));

    cooperative_groups::wait(block); // Joins all threads, waits for all copies to complete
    // Compute the index of the current batch for this block in global memory:
        // size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
        // size_t global_idx = block_batch_idx + threadIdx.x;
        // shared[local_idx] = global_in[global_idx];

        // block.sync(); // Wait for all copies to complete

        compute(global_out + block_batch_idx, shared, block.size()); // Compute and write result to global memory

        block.sync(); 
    }
}

// 性能测试函数
void benchmark_memcpy_async(size_t total_size, size_t batch_size, int block_size) {
    // 分配主机内存（使用固定内存以获得最佳性能）
    int* h_input = nullptr;
    int* h_output_async = nullptr;
    int* h_output_traditional = nullptr;
    
    cudaMallocHost(&h_input, total_size * sizeof(int));
    cudaMallocHost(&h_output_async, total_size * sizeof(int));
    cudaMallocHost(&h_output_traditional, total_size * sizeof(int));
    
    // 初始化数据
    for (size_t i = 0; i < total_size; ++i) {
        h_input[i] = static_cast<int>(i);
    }
    
    // 分配设备内存
    int* d_input = nullptr;
    int* d_output_async = nullptr;
    int* d_output_traditional = nullptr;
    float* d_kernel_time_async = nullptr;
    float* d_kernel_time_traditional = nullptr;
    
    cudaMalloc(&d_input, total_size * sizeof(int));
    cudaMalloc(&d_output_async, total_size * sizeof(int));
    cudaMalloc(&d_output_traditional, total_size * sizeof(int));
    cudaMalloc(&d_kernel_time_async, sizeof(float));
    cudaMalloc(&d_kernel_time_traditional, sizeof(float));
    
    // 计算网格和块尺寸
    size_t batch_sz = total_size / batch_size;
    dim3 block(block_size);
    dim3 grid((batch_size + block_size - 1) / block_size);
    
    size_t shared_mem_size = block_size * sizeof(int);
    cudaMemcpy(d_input, h_input, total_size * sizeof(int), cudaMemcpyHostToDevice);

    // 测试memcpy_async版本
    for (int i = 0; i < 20; ++i) {
        traditional_sync<<<grid, block, shared_mem_size>>>(d_output_async, d_input, total_size, batch_sz, d_kernel_time_async);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    traditional_sync<<<grid, block, shared_mem_size>>>(

        d_output_async, d_input, total_size, batch_sz, d_kernel_time_async);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_async, d_output_async, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    auto async_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    for (int i = 0; i < 20; ++i) {
    with_memcpy_async<<<grid, block, shared_mem_size>>>(
        d_output_traditional, d_input, total_size, batch_sz, d_kernel_time_traditional);    }
    // 测试传统同步版本
    start_time = std::chrono::high_resolution_clock::now();
    
    with_memcpy_async<<<grid, block, shared_mem_size>>>(
        d_output_traditional, d_input, total_size, batch_sz, d_kernel_time_traditional);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_traditional, d_output_traditional, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    auto traditional_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    
    // 验证结果正确性
    bool results_match = true;
    for (size_t i = 0; i < total_size; ++i) {
        if (h_output_async[i] != h_output_traditional[i]) {
            printf("结果不匹配: h_output_async[%zu] = %d, h_output_traditional[%zu] = %d\n", i, h_output_async[i], i, h_output_traditional[i]);
            results_match = false;
            // break;
        }
    }
    
    // 打印性能结果
    std::cout << "=== 性能测试结果 ===" << std::endl;
    std::cout << "数据大小: " << total_size << " 个整数 (" << total_size * sizeof(int) / 1024.0 << " KB)" << std::endl;
    std::cout << "批次大小: " << batch_size << ", 块大小: " << block_size << std::endl;
    std::cout << "批次数: " << batch_sz << std::endl;
    std::cout << std::endl;
    
    std::cout << "memcpy_async版本:" << std::endl;
    std::cout << "  - 总执行时间: " << async_duration.count() << " μs" << std::endl;
    // std::cout << "  - 内核执行时间: " << async_kernel_time << " ms" << std::endl;
    std::cout << "  - 总吞吐量: " << (total_size * sizeof(int) / 1024.0 / 1024.0) / (async_duration.count() / 1000000.0) << " MB/s" << std::endl;
    
    std::cout << "传统同步版本:" << std::endl;
    std::cout << "  - 总执行时间: " << traditional_duration.count() << " μs" << std::endl;
    // std::cout << "  - 内核执行时间: " << traditional_kernel_time << " ms" << std::endl;
    std::cout << "  - 总吞吐量: " << (total_size * sizeof(int) / 1024.0 / 1024.0) / (traditional_duration.count() / 1000000.0) << " MB/s" << std::endl;
    std::cout << std::endl;
    
    std::cout << "性能提升:" << std::endl;
    // std::cout << "  - 内核时间加速比: " << traditional_kernel_time / async_kernel_time << "x" << std::endl;
    std::cout << "  - 总时间加速比: " << traditional_duration.count() / (double)async_duration.count() << "x" << std::endl;
    std::cout << "  - 结果正确性: " << (results_match ? "✓ 通过" : "✗ 失败") << std::endl;
    
    // 清理资源
    cudaFreeHost(h_input);
    cudaFreeHost(h_output_async);
    cudaFreeHost(h_output_traditional);
    cudaFree(d_input);
    cudaFree(d_output_async);
    cudaFree(d_output_traditional);
    cudaFree(d_kernel_time_async);
    cudaFree(d_kernel_time_traditional);
}
int main() {
    // 设置不同的测试场景
    std::vector<std::tuple<size_t, size_t, int>> test_cases = {
        // {1024 * 1024, 4096, 256}   // 中等数据量，中等批次
        {2048 * 2048, 8192, 512}    // 大数据量，大批次
        // {512 * 512, 2048, 128}   ,    // 小数据量，小批次
        // {512 * 32, 32, 32}       // 小数据量，小批次
    };
    
    std::cout << "开始memcpy_async性能测试..." << std::endl;
    std::cout << "==========================================" << std::endl;
    
    for (const auto& test_case : test_cases) {
        size_t total_size = std::get<0>(test_case);
        size_t batch_size = std::get<1>(test_case);
        int block_size = std::get<2>(test_case);
        
        benchmark_memcpy_async(total_size, batch_size, block_size);
        std::cout << "==========================================" << std::endl;
    }
    
    return 0;
}