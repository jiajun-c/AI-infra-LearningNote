#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel: 简单的并行加法
template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int size) 
{
    // 计算当前线程的全局索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// C++ 调用入口函数
void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    // 检查输入必须在 CUDA 上
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    
    // 检查形状连续性 (Contiguous)
    // 实际项目中你可以处理非连续情况，这里为了演示简单要求连续
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    const int size = a.numel();
    
    // 线程块配置
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    // 启动 Kernel
    // 使用 AT_DISPATCH 宏自动处理 float/double 类型分发
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "vector_add_cuda", ([&] {
        vector_add_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            size
        );
    }));

    // 检查 Kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in vector_add_kernel: %s\n", cudaGetErrorString(err));
    }
}

// Python 绑定定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &vector_add_cuda, "Vector Add CUDA Kernel");
}