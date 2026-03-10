#include <cuda_runtime.h>
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// 核心 Kernel: 按照线程 ID 连续排布写回
// ============================================================================
template <typename T>
__global__ void inspect_partition_kernel(T const* in_ptr, T* out_ptr, int N) {
    // 1. 定义全局输入 Tensor
    Tensor g_in = make_tensor(make_gmem_ptr(in_ptr), make_shape(N));
    Tensor g_out = make_tensor(make_gmem_ptr(out_ptr), make_shape(N));

    Tensor b_in = local_tile(g_in, make_shape(N), make_coord(blockIdx.x));
    Tensor b_out = local_tile(g_out, make_shape(N), make_coord(blockIdx.x));

    auto layout = make_layout(make_shape(Int<256>{}));
    auto thr_in = local_partition(b_in, layout,threadIdx.x);
    auto thr_out = local_partition(b_out, layout,threadIdx.x);

    int out_offset = blockIdx.x * 1024 + threadIdx.x * size(thr_in); // 加上 block 偏移
    
    for (int i = 0; i < size(thr_in); i++) {
        out_ptr[out_offset + i] = thr_in(i);
    }
    // copy(thr_in, thr_out);
}

// ============================================================================
// 主函数验证与可视化
// ============================================================================
int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // 输入数据直接用 0 到 1023 的索引值，方便追踪每个元素去哪了
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
        h_out[i] = -1.0f; // 初始化为 -1
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 启动 1 个包含 256 线程的 Block
    dim3 block(256); 
    dim3 grid(1);

    inspect_partition_kernel<float><<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 打印前几个线程分到的数据
    std::cout << "=== local_partition 数据分发透视表 ===" << std::endl;
    for (int tid = 0; tid < 4; ++tid) {
        std::cout << "Thread " << tid << " 拿到的 4 个元素: [ ";
        for (int i = 0; i < 4; ++i) {
            std::cout << h_out[tid * 4 + i] << " ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "..." << std::endl;
    
    // 打印最后几个线程分到的数据
    for (int tid = 254; tid < 256; ++tid) {
        std::cout << "Thread " << tid << " 拿到的 4 个元素: [ ";
        for (int i = 0; i < 4; ++i) {
            std::cout << h_out[tid * 4 + i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    return 0;
}