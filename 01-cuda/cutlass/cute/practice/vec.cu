#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

// 错误检查宏
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
// 修复后的 Kernel
// ============================================================================
__global__ void vec_kernel(float* in, float* out, int N) {
    // 1. 包装全局 Tensor (动态长度 N)
    auto gIn = make_tensor(make_gmem_ptr(in), make_shape(N));
    auto gOut = make_tensor(make_gmem_ptr(out), make_shape(N));

    // 2. 切分给当前 Block (每块 1024 个元素)
    auto coord = make_coord(blockIdx.x);
    auto b_in = local_tile(gIn, make_shape(Int<1024>{}), coord);
    auto b_out = local_tile(gOut, make_shape(Int<1024>{}), coord);

    // 3. 定义 Thread 的 Layout (修复：必须是 make_layout)
    auto thr_layout = make_layout(Int<256>{});

    // 4. 将 Block 的数据切分给 Thread
    auto thr_in = local_partition(b_in, thr_layout, threadIdx.x);
    auto thr_out = local_partition(b_out, thr_layout, threadIdx.x);

    // 5. 遍历并计算
    #pragma unroll // 强烈建议加上：因为 size(thr_in) 在编译期已知为 4，展开后性能更好
    for (int i = 0; i < size(thr_in); i++) {
        thr_out(i) = thr_in(i) * 2.0f;
    }
}

// ============================================================================
// 主测试函数
// ============================================================================
int main() {
    // 故意设为一个不是 1024 倍数的 N，测试边界保护是否生效
    const int N = 10000; 
    size_t bytes = N * sizeof(float);

    // 分配并初始化 Host 内存
    std::vector<float> h_in(N);
    std::vector<float> h_out(N, 0.0f);
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // 分配 Device 内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // 配置线程模型
    // 线程块大小：256 线程
    // 网格大小：向上取整算出需要多少个 1024 大小的 Block
    int threads_per_block = 256;
    int elements_per_block = 1024;
    int blocks_per_grid = (N + elements_per_block - 1) / elements_per_block;

    std::cout << "Launching Kernel with Grid: " << blocks_per_grid 
              << ", Block: " << threads_per_block << std::endl;

    // 启动 Kernel
    vec_kernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷回结果
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    bool passed = true;
    for (int i = 0; i < N; i++) {
        float expected = h_in[i] * 2.0f;
        if (h_out[i] != expected) {
            std::cout << "Mismatch at index " << i 
                      << ": Expected " << expected << ", Got " << h_out[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "✓ Verification PASSED! Vector scaling works." << std::endl;
    }

    // 清理资源
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}