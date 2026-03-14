#include <iostream>
#include <iomanip>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

// 定义矩阵维度
constexpr int M = 16;
constexpr int N = 8;
constexpr int STRIDE = N; // 行优先 (Row-Major) 的 Stride

// ==============================================================================
// 1. 传统 CUDA 版本：手动计算坐标 (痛苦但直观)
// ==============================================================================
__global__ void traditional_kernel(float* g_out) {
    // 申请 16x8 的 Shared Memory
    __shared__ float smem[M * N];

    // 你的传统坐标映射逻辑
    int tid = threadIdx.x;
    int row = (tid / 8) * 4; 
    int col = tid % 8;

    // 为了填满 16x8，每个线程在这个列上连续处理 4 个元素
    for (int i = 0; i < 4; ++i) {
        float* my_ptr = smem + (row + i) * STRIDE + col;
        // 把线程 ID 写进去，方便我们在外围查看映射关系
        *my_ptr = static_cast<float>(tid);
    }

    __syncthreads();

    // 验证：把 Shared Memory 写回 Global Memory
    for (int i = 0; i < 4; ++i) {
        g_out[(row + i) * STRIDE + col] = smem[(row + i) * STRIDE + col];
    }
}

// ==============================================================================
// 2. CuTe 版本：留给你来写！
// ==============================================================================
__global__ void cute_kernel(float* g_out) {
    __shared__ float smem[M * N];
    auto layout = make_layout(make_shape(M, N), LayoutRight{});
    auto sG = make_tensor(make_smem_ptr(smem), layout);
    auto gG = make_tensor(make_gmem_ptr(g_out), layout);

    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, float>{},
        Layout<Shape<_4, _8>, Stride<_8, _1>>{},
        Layout<Shape<_4, _1>>{});
    int tid = threadIdx.x;
    auto thr_slice = tiled_copy.get_thread_slice(tid);

    auto tGsG = thr_slice.partition_S(sG);

    for (int i = 0; i < size(tGsG); i++) {
        tGsG(i) = static_cast<float>(tid);
    }

    __syncthreads();

    auto tGgG = thr_slice.partition_D(gG);
    copy(tiled_copy, tGsG, tGgG);
    
}

// ==============================================================================
// Host 端测试验证代码
// ==============================================================================
void print_matrix(const std::vector<float>& mat, const std::string& name) {
    std::cout << "=== " << name << " (16x8) ===" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(4) << mat[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int num_elements = M * N;
    size_t bytes = num_elements * sizeof(float);

    float *d_out_trad, *d_out_cute;
    cudaMalloc(&d_out_trad, bytes);
    cudaMalloc(&d_out_cute, bytes);
    cudaMemset(d_out_trad, 0, bytes);
    cudaMemset(d_out_cute, 0, bytes);

    // 启动 32 个线程 (1个 Warp)
    dim3 threads(32);
    dim3 blocks(1);

    std::cout << "🚀 Launching Traditional Kernel..." << std::endl;
    traditional_kernel<<<blocks, threads>>>(d_out_trad);
    cudaDeviceSynchronize();

    std::cout << "🚀 Launching CuTe Kernel..." << std::endl;
    cute_kernel<<<blocks, threads>>>(d_out_cute);
    cudaDeviceSynchronize();

    std::vector<float> h_out_trad(num_elements);
    std::vector<float> h_out_cute(num_elements);

    cudaMemcpy(h_out_trad.data(), d_out_trad, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_cute.data(), d_out_cute, bytes, cudaMemcpyDeviceToHost);

    // 打印传统版本的结果，让你直观看到 32 个线程是怎么铺在 16x8 矩阵上的
    print_matrix(h_out_trad, "Traditional Result");
    // 等你写完 CuTe 版本，可以放开下面这行对比结果
    print_matrix(h_out_cute, "CuTe Result");

    cudaFree(d_out_trad);
    cudaFree(d_out_cute);

    return 0;
}