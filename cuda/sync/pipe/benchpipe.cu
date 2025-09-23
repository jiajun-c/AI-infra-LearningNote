#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#pragma nv_diag_suppress static_var_with_dynamic_init

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 设备端计算函数：将共享内存中的数据翻倍并写入全局内存
__device__ void compute(int* global_out, int const* shared_in) {
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 简单示例：将数据翻倍
    for (int j = 0; j < 100; j++)
    global_out[ threadIdx.x] = shared_in[threadIdx.x]*shared_in[threadIdx.x]/3* 2*shared_in[threadIdx.x];
    //  = computed_val;
}
__global__ void baseline(int* global_out, int const* global_in, int size, int batch) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < 100; j++)
            global_out[idx + gridDim.x *blockDim.x*i ] = global_in[idx]* global_in[idx]*2/3*2* global_in[idx];

        }
}
// 内核函数：使用流水线技术进行异步数据拷贝和计算
__global__ void with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // 假设输入大小符合 batch_sz * grid_size

    constexpr size_t stages_count = 2; // 两阶段流水线
    // 两个批次必须适合共享内存：
    extern __shared__ int shared[];  // stages_count * block.size() * sizeof(int) 字节
    size_t shared_offset[stages_count] = { 0, block.size() }; // 每个批次的偏移量

    // 为两阶段cuda::pipeline分配共享存储：
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // 每个线程处理`batch_sz`个元素。
    // 计算此线程块的批次`batch`在全局内存中的偏移量：
    auto block_batch = [&](size_t batch) -> size_t {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    // 通过提交`memcpy_async`来获取块的整个批次，初始化第一个流水线阶段：
    if (batch_sz == 0) return;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shared + shared_offset[0], global_in + block_batch(0), sizeof(int) * block.size(), pipeline);
    pipeline.producer_commit();

    // 流水线拷贝/计算：
    for (size_t batch = 1; batch < batch_sz; ++batch) {
        // 计算和拷贝阶段的索引：
        size_t compute_stage_idx = (batch - 1) % 2;
        size_t copy_stage_idx = batch % 2;
        size_t global_idx = block_batch(batch);

        // 所有生产者线程集体获取流水线头阶段：
        pipeline.producer_acquire();

        // 将异步拷贝提交到流水线头阶段，以便在下一个循环迭代中计算
        cuda::memcpy_async(block, shared + shared_offset[copy_stage_idx], global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // 集体提交（推进）流水线头阶段
        pipeline.producer_commit();

        // 集体等待提交到先前`compute`阶段的操作完成：
        pipeline.consumer_wait();

        // 与"拷贝"阶段的memcpy_async重叠的计算：
        compute(global_out + block_batch(batch-1), shared + shared_offset[compute_stage_idx]);

        // 集体释放阶段资源
        pipeline.consumer_release();
    }

    // 计算最后一次迭代获取的数据
    pipeline.consumer_wait();
    compute(global_out + block_batch(batch_sz-1), shared + shared_offset[(batch_sz - 1) % 2]);
    pipeline.consumer_release();
}

int main() {
    // 设置问题规模
    cudaSetDevice(2);
    const size_t total_elements = 4096*4096*16;     // 总元素数
    const size_t batch_size = 16;            // 批次数
    const size_t elements_per_batch = total_elements / batch_size; // 每批元素数
    
    // 设置线程块和网格大小
    const int block_size = 256;             // 每个线程块256个线程
    const int grid_size = total_elements / block_size / batch_size; // 计算网格大小
    
    // 计算共享内存大小：两阶段，每阶段block_size个元素
    const size_t shared_mem_size = 2 * block_size * sizeof(int);
    
    std::cout << "配置参数:" << std::endl;
    std::cout << "  总元素数: " << total_elements << std::endl;
    std::cout << "  批次数: " << batch_size << std::endl;
    std::cout << "  每批元素数: " << elements_per_batch << std::endl;
    std::cout << "  线程块大小: " << block_size << std::endl;
    std::cout << "  网格大小: " << grid_size << std::endl;
    std::cout << "  共享内存大小: " << shared_mem_size << " 字节" << std::endl;
    
    // 分配和初始化主机内存
    std::vector<int> h_input(total_elements);
    std::vector<int> h_output(total_elements);
    std::vector<int> h_expected(total_elements);
    
    for (size_t i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<int>(i);
        h_expected[i] = h_input[i] * 2; // 预期结果：值翻倍
    }
    
    // 分配设备内存
    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, total_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, total_elements * sizeof(int)));
    
    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), 
                                total_elements * sizeof(int), 
                                cudaMemcpyHostToDevice));
    
    // 初始化输出设备内存
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, total_elements * sizeof(int)));
    
    // 启动内核
    std::cout << "启动内核..." << std::endl;
    
    // 检查设备是否支持协作组
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    
    dim3 grid(grid_size);
    dim3 block(block_size);
    // int *d_input1, *d_output1;
    // float elapsedTime1 = 0.0;
    // CHECK_CUDA_ERROR(cudaMalloc(&d_input1, total_elements * sizeof(int)));
    // CHECK_CUDA_ERROR(cudaMalloc(&d_output1, total_elements * sizeof(int)));
    // cudaEvent_t start1, stop1;
    // cudaEventCreate(&start1);
    // cudaEventCreate(&stop1);
    // cudaEventRecord(start1, 0);
    // baseline<<<grid, block>>>(d_output1, d_input1, total_elements,batch_size);
    // cudaEventRecord(stop1, 0);
    // cudaEventSynchronize(stop1);

    // cudaEventElapsedTime(&elapsedTime1, start1, stop1);
    // std::cout << elapsedTime1 << std::endl;


        cudaEvent_t start, stop;
        float elapsedTime = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // 回退到传统启动方式
        with_staging<<<grid, block, shared_mem_size>>>(
            d_output, d_input, total_elements, batch_size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << elapsedTime << std::endl;


    // 检查内核启动错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 等待内核完成
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 将结果复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, 
                                total_elements * sizeof(int), 
                                cudaMemcpyDeviceToHost));

    // 验证结果
    bool success = true;
    for (size_t i = 0; i < total_elements; ++i) {
        if (h_output[i] != h_expected[i]) {
            std::cerr << "结果验证失败于索引 " << i 
                      << ": 期望 " << h_expected[i] 
                      << ", 得到 " << h_output[i] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "结果验证成功! 所有值都正确翻倍。" << std::endl;
        
        // 显示前10个结果作为示例
        std::cout << "前10个结果:" << std::endl;
        for (int i = 0; i < 10 && i < total_elements; ++i) {
            std::cout << "  输入[" << i << "] = " << h_input[i] 
                      << ", 输出[" << i << "] = " << h_output[i] << std::endl;
        }
    }
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);
    
    return success ? 0 : 1;
}