#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void grid_sync_kernel(int *data) {
    // 获取当前线程的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 获取网格组对象
    cg::grid_group grid = cg::this_grid();
    if (tid == 0) {
        printf("num_threads: %d num_blocks: %d\n", grid.num_threads(), grid.num_blocks());
    }
    // 第一阶段：每个线程处理数据
    data[tid] = data[tid] * 2;
    
    // 网格同步：所有线程必须到达此点才能继续
    grid.sync();
    
    // 第二阶段：所有线程同步后继续处理
    data[tid] = data[tid] + 1;
}

int main() {
    const int data_size = 1024;
    const int threads_per_block = 256;
    const int blocks_per_grid = (data_size + threads_per_block - 1) / threads_per_block;
    int dev = 0;
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, dev);
    int cooperativeSupport = 0;
    cudaDeviceGetAttribute(&cooperativeSupport, cudaDevAttrCooperativeLaunch, dev);
    if (!cooperativeSupport) {
        printf("Cooperative kernel launch is not supported on this device.\n");
        return -1;
    }
    printf("Device supports cooperative launch: Yes\n");
    // 检查计算能力
    if (prop.major < 9) {
        printf("Device compute capability (%d.%d) is insufficient. Requires >= 6.0\n", prop.major, prop.minor);
        return -1;
    }
    int *h_data = new int[data_size];
    int *d_data;
    
    // 初始化数据
    for (int i = 0; i < data_size; i++) {
        h_data[i] = i;
    }
    
    // 分配设备内存
    cudaMalloc(&d_data, data_size * sizeof(int));
    cudaMemcpy(d_data, h_data, data_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // 计算合适的网格大小以确保所有块能同时驻留GPU
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, grid_sync_kernel, threads_per_block, 0);
    
    int num_blocks = num_sms * max_blocks_per_sm;
    printf("num_blocks %d\n", num_blocks);
    // 使用协作启动API启动内核
    void *kernel_args[] = { &d_data };
    cudaLaunchCooperativeKernel((void*)grid_sync_kernel, num_blocks, threads_per_block, kernel_args);
    
    // 等待内核完成
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(h_data, d_data, data_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %d (expected: %d)\n", i, h_data[i], (i * 2) + 1);
    }
    
    // 清理资源
    delete[] h_data;
    cudaFree(d_data);
    
    return 0;
}