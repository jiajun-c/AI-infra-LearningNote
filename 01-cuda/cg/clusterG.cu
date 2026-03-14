#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// 核函数：演示cluster_group的基本用法
__global__ void cluster_kernel(int *global_data) {
    // 获取当前线程所在的集群组
    cg::cluster_group cluster = cg::this_cluster();
    
    // 获取线程在集群中的全局排名
    unsigned int cluster_rank = cluster.thread_rank();
    unsigned int block_rank = cluster.block_rank(); // 当前块在集群中的排名
    
    printf("cluster_rank %d %d\n", cluster_rank, block_rank);
    // 每个线程块选择一个代表线程（例如rank 0）来加载数据
    __shared__ int local_data;
    cg::thread_block block = cg::this_thread_block();
    
    if (block.thread_rank() == 0) {
        // 从全局内存加载数据到共享内存
        local_data = global_data[block_rank];
    }
    block.sync(); // 块内同步，确保所有线程看到共享数据

    // 进行一些计算（示例：简单乘以线程索引）
    int computed_value = cluster_rank;

    // 集群级别的同步：确保所有线程块都到达此点
    cluster.sync();

    // 将计算结果存回全局内存
    global_data[cluster_rank] = computed_value;
}

int main() {
    int num_blocks = 4;  // 集群中的线程块数量
    int threads_per_block = 32;
    int data_size = num_blocks * threads_per_block;
    
    int *h_data, *d_data;
    
    // 分配主机内存
    h_data = (int*)malloc(data_size * sizeof(int));
    
    // 初始化数据
    for (int i = 0; i < data_size; ++i) {
        h_data[i] = 0; // 示例数据
    }
    
    // 分配设备内存
    cudaMalloc((void**)&d_data, data_size * sizeof(int));
    
    // 将数据复制到设备
    cudaMemcpy(d_data, h_data, data_size * sizeof(int), cudaMemcpyHostToDevice);

    // 设置网格和块维度
    dim3 grid_dim(num_blocks);
    dim3 block_dim(threads_per_block);

    // 注意：使用cluster需要特殊的启动方式
    // 此示例假设使用支持cluster的架构（如计算能力9.0+）
    // 实际使用时需要查询设备属性和使用cudaLaunchCooperativeKernel
    cluster_kernel<<<grid_dim, block_dim>>>(d_data);
    
    // 等待核函数完成
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(h_data, d_data, data_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 打印部分结果验证
    printf("First 10 results:\n");
    for (int i = 0; i < 4*32; ++i) {
        printf("%d: %d\n", i, h_data[i]);
    }
    
    // 释放资源
    free(h_data);
    cudaFree(d_data);
    
    return 0;
}