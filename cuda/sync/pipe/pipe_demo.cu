#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <iostream>
#include <cassert>

// 禁用 `pipeline_shared_state` 初始化警告
#pragma nv_diag_suppress static_var_with_dynamic_init

// 示例计算函数：将共享内存中的数据翻倍
template <typename T>
__device__ void compute(T* ptr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ptr[idx] = 0; // 实际应用中替换为实际计算逻辑
}

// 内核函数定义（与您提供的完全一致）
template <typename T>
__global__ void example_kernel(T* global0, T* global1, cuda::std::size_t subset_count) {
    extern __shared__ T s[];
    auto group = cooperative_groups::this_thread_block();
    T* shared[2] = { s, s + 2 * group.size() };

    // 创建流水线
    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(group, &shared_state);

    // 初始化流水线
    pipeline.producer_acquire();
    cuda::memcpy_async(group, shared[0],
                       &global0[0], sizeof(T) * group.size(), pipeline);
    cuda::memcpy_async(group, shared[0] + group.size(),
                       &global1[0], sizeof(T) * group.size(), pipeline);
    pipeline.producer_commit();

    // 流水线处理循环
    for (cuda::std::size_t subset = 1; subset < subset_count; ++subset) {
        pipeline.producer_acquire();
        cuda::memcpy_async(group, shared[subset % 2],
                           &global0[subset * group.size()],
                           sizeof(T) * group.size(), pipeline);
        cuda::memcpy_async(group, shared[subset % 2] + group.size(),
                           &global1[subset * group.size()],
                           sizeof(T) * group.size(), pipeline);
        pipeline.producer_commit();
        pipeline.consumer_wait();
        compute(shared[(subset - 1) % 2]);
        pipeline.consumer_release();
    }

    // 处理最后一批数据
    pipeline.consumer_wait();
    compute(shared[(subset_count - 1) % 2]);
    pipeline.consumer_release();
}
template void __global__ example_kernel<int>(int*, int*, cuda::std::size_t);

// 主机端辅助函数：检查CUDA错误
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    cudaSetDevice(2);
    // 设置问题规模
    const size_t total_elements = 1024; // 总数据元素数
    const size_t subset_count = 4;      // 数据子集数
    const size_t elements_per_subset = total_elements / subset_count; // 每个子集元素数

    // 设置线程块和网格大小
    const dim3 block_dim(256); // 每个线程块256个线程
    const dim3 grid_dim(1);    // 使用1个线程块（可根据需要调整）

    // 计算共享内存大小：双缓冲，每个缓冲区大小 = 2 * block_dim.x * sizeof(int)
    const size_t shared_mem_size = 4 * block_dim.x * sizeof(int);

    // 分配和初始化主机内存
    int* h_global0 = new int[total_elements];
    int* h_global1 = new int[total_elements];
    for (size_t i = 0; i < total_elements; ++i) {
        h_global0[i] = static_cast<int>(i);
        h_global1[i] = static_cast<int>(i * 2);
    }

    // 分配设备内存
    int *d_global0, *d_global1;
    CHECK_CUDA_ERROR(cudaMalloc(&d_global0, total_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global1, total_elements * sizeof(int)));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_global0, h_global0, total_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_global1, h_global1, total_elements * sizeof(int), cudaMemcpyHostToDevice));

    // 启动内核
    std::cout << "启动内核: grid_dim = (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), "
              << "block_dim = (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << "), "
              << "shared_mem_size = " << shared_mem_size << " bytes" << std::endl;

    example_kernel<int><<<grid_dim, block_dim, shared_mem_size>>>(
        d_global0, d_global1, subset_count
    );

    // 检查内核启动错误
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 等待内核完成
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 将结果复制回主机（可选，取决于是否需要查看结果）
    CHECK_CUDA_ERROR(cudaMemcpy(h_global0, d_global0, total_elements * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_global1, d_global1, total_elements * sizeof(int), cudaMemcpyDeviceToHost));

    // 验证结果（简单示例）
    std::cout << "内核执行完成。检查前10个元素：" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_global0[" << i << "] = " << h_global0[i] << ", "
                  << "h_global1[" << i << "] = " << h_global1[i] << std::endl;
    }

    // 释放资源
    delete[] h_global0;
    delete[] h_global1;
    cudaFree(d_global0);
    cudaFree(d_global1);

    return 0;
}