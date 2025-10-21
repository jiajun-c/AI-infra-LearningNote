#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <iostream>
#include <cuda_runtime.h>

// 禁用 `pipeline_shared_state` 初始化警告
#pragma nv_diag_suppress static_var_with_dynamic_init

// 简单的计算函数示例：对共享内存中的两个数据块进行逐元素相加
template <typename T>
__device__ void compute(T* shared_block) {
    int tid = cooperative_groups::this_thread_block().thread_rank();
    int block_size = cooperative_groups::this_thread_block().size();
    // 假设每个共享内存块包含block_size个元素，计算两个部分的相加
    if (tid < block_size) {
        shared_block[tid] += shared_block[tid + block_size];
    }
    __syncthreads(); // 确保块内线程同步
}

template <typename T>
__global__ void baseline_kernel(T* global0, T* global1, T* output, cuda::std::size_t subset_count, int data_size_per_subset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size_per_subset) {
        return;
    }
    for (int i = 0; i < subset_count; ++i) {
        output[i * data_size_per_subset + idx] = global0[i * data_size_per_subset + idx] + global1[i * data_size_per_subset + idx];
    }
}
// 使用CUDA Pipeline的主内核函数
template <typename T>
__global__ void pipeline_kernel(T* global0, T* global1, T* output, cuda::std::size_t subset_count, int data_size_per_subset) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_raw[];
    T* shared_base = reinterpret_cast<T*>(shared_memory_raw);
    
    auto group = cooperative_groups::this_thread_block();
    int block_size = group.size();
    // 为两个流水线阶段分配共享内存
    T* shared_stages[2] = { shared_base, shared_base + 2 * block_size };

    // 创建Pipeline对象
    constexpr auto scope = cuda::thread_scope_block;
    constexpr cuda::std::size_t stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(group, &shared_state);

    // 初始化Pipeline：预加载第一个数据子集
    pipeline.producer_acquire();
    cuda::memcpy_async(group, shared_stages[0], 
                       &global0[0], sizeof(T) * block_size, pipeline);
    cuda::memcpy_async(group, shared_stages[0] + block_size,
                       &global1[0], sizeof(T) * block_size, pipeline);
    pipeline.producer_commit();

    // 主循环：重叠数据加载与计算
    for (cuda::std::size_t subset = 1; subset < subset_count; ++subset) {
        // 生产者阶段：异步加载下一个数据子集
        pipeline.producer_acquire();
        cuda::memcpy_async(group, shared_stages[subset % 2],
                           &global0[subset * block_size],
                           sizeof(T) * block_size, pipeline);
        cuda::memcpy_async(group, shared_stages[subset % 2] + block_size,
                           &global1[subset * block_size],
                           sizeof(T) * block_size, pipeline);
        pipeline.producer_commit();

        // 消费者阶段：处理已加载的上一个子集
        pipeline.consumer_wait();
        compute(shared_stages[(subset - 1) % 2]);  // 执行计算
        // 将结果写回全局内存（可选，根据实际需求调整）
        if (threadIdx.x < block_size) {
            output[(subset - 1) * block_size + threadIdx.x] = shared_stages[(subset - 1) % 2][threadIdx.x];
        }
        pipeline.consumer_release();
    }

    // 处理管道中剩余的最后一批数据
    pipeline.consumer_wait();
    compute(shared_stages[(subset_count - 1) % 2]);
    if (threadIdx.x < block_size) {
        output[(subset_count - 1) * block_size + threadIdx.x] = shared_stages[(subset_count - 1) % 2][threadIdx.x];
    }
    pipeline.consumer_release();
}

int main() {
    const int TOTAL_ELEMENTS = 1024*1024*1024;  // 总数据元素数量
    const int BLOCK_SIZE = 256;       // 线程块大小
    const int SUBSET_COUNT = TOTAL_ELEMENTS / BLOCK_SIZE;  // 子集数量
    const int ELEMENTS_PER_SUBSET = BLOCK_SIZE;

    // 分配主机内存
    int *h_global0, *h_global1, *h_output;
    h_global0 = new int[TOTAL_ELEMENTS];
    h_global1 = new int[TOTAL_ELEMENTS];
    h_output = new int[TOTAL_ELEMENTS];

    // 初始化主机数据
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        h_global0[i] = i;
        h_global1[i] = i * 2;
    }

    // 分配设备内存
    int *d_global0, *d_global1, *d_output;
    cudaMalloc(&d_global0, TOTAL_ELEMENTS * sizeof(int));
    cudaMalloc(&d_global1, TOTAL_ELEMENTS * sizeof(int));
    cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(int));

    // 将数据复制到设备
    cudaMemcpy(d_global0, h_global0, TOTAL_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global1, h_global1, TOTAL_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    // 计算内核启动参数
    dim3 block(BLOCK_SIZE);
    dim3 grid(1);  // 使用1个线程块
    size_t shared_mem_size = 2 * 2 * BLOCK_SIZE * sizeof(int);  // 双缓冲，每个阶段2*BLOCK_SIZE个元素

    // 启动内核
    for (int i = 0; i < 10; i++) {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        baseline_kernel<int><<<grid, block, shared_mem_size>>>(d_global0, d_global1, d_output, SUBSET_COUNT, ELEMENTS_PER_SUBSET);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        cudaEventRecord(stop);
        float elapsedTimes;
        cudaEventElapsedTime(&elapsedTimes, start, stop);
 
        // Output results
 
        std::cout << " time" << ": " << elapsedTimes << " ms" << std::endl;
    }

    // 检查内核启动错误
    cudaError_t kernelLaunchStatus = cudaGetLastError();
    if (kernelLaunchStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelLaunchStatus) << std::endl;
        return -1;
    }

    // 等待设备计算完成
    cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
        std::cerr << "Device synchronization failed: " << cudaGetErrorString(syncStatus) << std::endl;
        return -1;
    }

    // 将结果复制回主机
    cudaMemcpy(h_output, d_output, TOTAL_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证结果（简单检查前10个元素）
    std::cout << "Checking first 10 results (expected: input0 + input1):" << std::endl;
    for (int i = 0; i < 10 && i < TOTAL_ELEMENTS; i++) {
        int expected = h_global0[i] + h_global1[i];  // 根据compute函数逻辑
        std::cout << "Index " << i << ": " << h_output[i] << " (expected: " << expected << ")" << std::endl;
    }

    // 释放资源
    cudaFree(d_global0);
    cudaFree(d_global1);
    cudaFree(d_output);
    delete[] h_global0;
    delete[] h_global1;
    delete[] h_output;

    std::cout << "Pipeline example completed successfully!" << std::endl;
    return 0;
}