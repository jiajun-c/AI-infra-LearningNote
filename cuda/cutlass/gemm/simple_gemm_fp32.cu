#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 必须包含 CUTLASS 的核心 GEMM 头文件
// 编译时需添加 include 路径: -I/path/to/cutlass/include
#include "cutlass/gemm/device/gemm.h"

// 宏检查 CUDA 错误
#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        return -1; \
    }

int main() {
    // 1. 定义问题规模
    int M = 128;
    int N = 128;
    int K = 128;
    float alpha = 1.0f;
    float beta = 0.0f;

    // 2. 定义 CUTLASS GEMM 类型
    // <元素类型A, 布局A, 元素类型B, 布局B, 元素类型C, 布局C>
    using Layout = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        float, Layout,  // Matrix A
        float, Layout,  // Matrix B
        float, Layout   // Matrix C & Output
    >;

    // 3. 分配 GPU 内存 (标准 CUDA 操作)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 初始化数据 (简单填充，实际使用时请拷贝真实数据)
    CUDA_CHECK(cudaMemset(d_A, 1, size_A)); // A 全为 0 (这里简单memset可能导致NaN, 仅演示流程)
    CUDA_CHECK(cudaMemset(d_B, 1, size_B)); // B 全为 0
    CUDA_CHECK(cudaMemset(d_C, 0, size_C)); // C 全为 0

    // 4. 配置 CUTLASS 参数
    // 列主序的 Stride (Leading Dimension) 通常是行数
    int lda = M;
    int ldb = K;
    int ldc = M;

    Gemm::Arguments args(
        {M, N, K},          // Problem Size
        {d_A, lda},         // Tensor A ref
        {d_B, ldb},         // Tensor B ref
        {d_C, ldc},         // Tensor C ref (Accumulator)
        {d_C, ldc},         // Tensor D ref (Output, 通常与 C 相同)
        {alpha, beta}       // Epilogue scalars
    );

    // 5. 实例化并运行
    Gemm gemm_op;
    
    // 这一步会进行 Kernel Launch
    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS kernel failed: " << std::endl;
        return -1;
    }

    // 同步等待执行完成
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "CUTLASS GEMM executed successfully!" << std::endl;

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
