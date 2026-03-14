#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

// 定义数据类型：float (4 bytes)
// 因为 4 个 float 刚好是 16 bytes = 128 bits，完美契合 uint128_t 的向量化指令
using TA = float;

// 
__global__ void one_thread_copy_kernel(const TA* g_in, TA* g_out, int M, int N) {
    // __shared__ TA smem[32*32];
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TA>{}, 
        Layout<Shape<_1, _1>, Stride<_1, _1>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape<_32,_32>>{}                   // Value  Layout: M-major (ColMajor)
    );
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        auto thr_copy = copyA.get_thread_slice(threadIdx.x);
        Tensor gA_src = make_tensor(make_gmem_ptr(g_in), make_shape(M, N));
        Tensor gA_dst = make_tensor(make_gmem_ptr(g_out), make_shape(M, N)); 
        Tensor tAgA_src = thr_copy.partition_S(gA_src);
        Tensor tAgA_dst = thr_copy.partition_D(gA_dst);
        copy(copyA, tAgA_src, tAgA_dst);
    }
}

int main() {
    // 假设我们要处理一个 32 x 32 的矩阵
    int M = 32;
    int N = 32;
    int num_elements = M * N;

    std::vector<TA> h_in(num_elements);
    std::vector<TA> h_out(num_elements, 0.0f);

    // 初始化测试数据
    for (int i = 0; i < num_elements; ++i) {
        h_in[i] = static_cast<TA>(i);
    }

    TA *d_in, *d_out;
    cudaMalloc(&d_in, num_elements * sizeof(TA));
    cudaMalloc(&d_out, num_elements * sizeof(TA));
    cudaMemcpy(d_in, h_in.data(), num_elements * sizeof(TA), cudaMemcpyHostToDevice);

    dim3 threads(32);
    dim3 blocks(M / 4, N / 8);

    std::cout << "🚀 Launching CuTe Cooperative Copy Kernel..." << std::endl;
    one_thread_copy_kernel<<<blocks, threads>>>(d_in, d_out, M, N);
    cudaDeviceSynchronize();

    // 检查是否有 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaMemcpy(h_out.data(), d_out, num_elements * sizeof(TA), cudaMemcpyDeviceToHost);

    // 简单验证结果
    bool success = true;
    for (int i = 0; i < num_elements; ++i) {
        if (h_in[i] != h_out[i]) {
            std::cerr << "Mismatch at index " << i << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "✅ TiledCopy execution verified successfully!" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}