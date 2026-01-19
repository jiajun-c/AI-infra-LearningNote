#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include <iostream>
#include <vector>
#include <cute/tensor.hpp>
using namespace cute;

__global__ void create_tensor_kernel(float* global_ptr) {
    auto layout = make_layout(make_shape(2, 2, 2));
    auto tensor = make_tensor(make_gmem_ptr(global_ptr), layout);
    __shared__ float smem[64];
    auto smemLayout = make_layout(make_shape(4, 4, 4));
    auto tensor_smem = make_tensor(make_smem_ptr(smem), smemLayout);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 使用 CuTe 内置的 print 函数
        printf("Layout info:\n");
        print(layout); 
        printf("\n\nTensor info:\n");
        print(tensor);
        printf("\n");
        
        // 演示：访问 Tensor 的第 0 个元素
        // 语法: tensor(坐标)
        printf("Element at index 0: %f\n", tensor(0, 1, 1));
    }
}

int main() {
    // 1. 在 Host 端分配显存
    float* d_ptr;
    size_t num_elements = 8;
    size_t size_bytes = num_elements * sizeof(float);
    
    cudaError_t err = cudaMalloc(&d_ptr, size_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 初始化一些数据以便观察 (可选)
    float h_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    cudaMemcpy(d_ptr, h_data, size_bytes, cudaMemcpyHostToDevice);

    // 2. 启动 Kernel
    create_tensor_kernel<<<1, 1>>>(d_ptr);
    cudaDeviceSynchronize();

    // 3. 释放资源
    cudaFree(d_ptr);

    return 0;
}