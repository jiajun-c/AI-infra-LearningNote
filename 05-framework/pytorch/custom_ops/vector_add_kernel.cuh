#pragma once
#include <cuda_runtime.h>

// 简单的 vector_add kernel，两种绑定方式共用同一个 kernel
// 重点不在 kernel 本身，而在绑定层如何处理 stream
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host-side launcher，接受 cudaStream_t 参数
// 这是 CUDA Graph 兼容性的关键：调用方需要传入正确的 stream
inline void launch_vector_add(const float* a, const float* b, float* c, int N,
                              cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, N);
}
