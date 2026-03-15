#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUTLASS 核心头文件 (基于 CUTLASS 2.x Device API)
#include "cutlass/gemm/device/gemm.h"

// 宏：检查 CUDA 运行时错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 宏：检查 cuBLAS 错误
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " - status code " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

///////////////////////////////////////////////////////////////////////////////////////////////////

// 1. 定义 CutlassSgemmNN 类型
// <ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
using ColumnMajor = cutlass::layout::ColumnMajor;
using CutlassSgemmNN = cutlass::gemm::device::Gemm<
    float, ColumnMajor,  // 矩阵 A: 单精度, 列主序 (N)
    float, ColumnMajor,  // 矩阵 B: 单精度, 列主序 (N)
    float, ColumnMajor   // 矩阵 C/D: 单精度, 列主序
>;

///////////////////////////////////////////////////////////////////////////////////////////////////

// cuBLAS SGEMM benchmark
// cuBLAS 原生就是列主序，与 CUTLASS ColumnMajor 完全一致
// C = alpha * A * B + beta * C
// A(M,K) ColMajor lda=M, B(K,N) ColMajor ldb=K, C(M,N) ColMajor ldc=M
float benchmark_cublas_sgemm(float *d_A, float *d_B, float *d_C_cublas,
                             int M, int N, int K,
                             float alpha, float beta,
                             int warmup_iters, int bench_iters) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 设置使用 Tensor Core (如果可用，cublas 会自动选择最优实现)
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // NoTrans, NoTrans
                                 M, N, K,
                                 &alpha,
                                 d_A, M,     // lda = M
                                 d_B, K,     // ldb = K
                                 &beta,
                                 d_C_cublas, M));  // ldc = M
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K,
                                 &alpha,
                                 d_A, M,
                                 d_B, K,
                                 &beta,
                                 d_C_cublas, M));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / bench_iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    return avg_ms;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
    // 2. 定义测试的问题规模
    int M = 4096;
    int N = 4096;
    int K = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;
    int warmup_iters = 10;
    int bench_iters = 100;

    // 打印 GPU 信息
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Benchmarking SGEMM (M=" << M << ", N=" << N << ", K=" << K << ")\n";
    std::cout << "Warmup=" << warmup_iters << ", Iterations=" << bench_iters << "\n\n";

    // 3. 分配 GPU 内存
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C_cutlass, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C_cutlass, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, size_C));

    // 用随机数据初始化，避免全零导致编译器/硬件走特殊路径
    {
        std::vector<float> h_A((size_t)M * K), h_B((size_t)K * N);
        srand(2023);
        for (auto& v : h_A) v = static_cast<float>(rand() % 100) / 100.0f - 0.5f;
        for (auto& v : h_B) v = static_cast<float>(rand() % 100) / 100.0f - 0.5f;
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_C_cutlass, 0, size_C));
    CUDA_CHECK(cudaMemset(d_C_cublas, 0, size_C));

    // =========================================================================
    //  CUTLASS SGEMM Benchmark
    // =========================================================================
    int lda = M;
    int ldb = K;
    int ldc = M;

    CutlassSgemmNN::Arguments args(
        {M, N, K},              // Problem Size
        {d_A, lda},             // Tensor A
        {d_B, ldb},             // Tensor B
        {d_C_cutlass, ldc},     // Tensor C (source)
        {d_C_cutlass, ldc},     // Tensor D (output)
        {alpha, beta}           // Epilogue scalars
    );

    CutlassSgemmNN gemm_op;

    // 检查硬件是否支持该配置
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
        std::cerr << "This CUTLASS GEMM config is not supported on this device!" << std::endl;
        return -1;
    }

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        gemm_op(args);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark CUTLASS
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        gemm_op(args);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cutlass_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cutlass_total_ms, start, stop));
    float cutlass_avg_ms = cutlass_total_ms / bench_iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // =========================================================================
    //  cuBLAS SGEMM Benchmark
    // =========================================================================
    float cublas_avg_ms = benchmark_cublas_sgemm(
        d_A, d_B, d_C_cublas, M, N, K, alpha, beta, warmup_iters, bench_iters);

    // =========================================================================
    //  计算 TFLOPS 并打印对比结果
    // =========================================================================
    double flops = 2.0 * M * N * K;
    double cutlass_tflops = (flops / (cutlass_avg_ms * 1e-3)) / 1e12;
    double cublas_tflops  = (flops / (cublas_avg_ms * 1e-3)) / 1e12;

    std::cout << "=============================================================\n";
    std::cout << "  SGEMM Performance Comparison (M=" << M
              << " N=" << N << " K=" << K << ")\n";
    std::cout << "=============================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  CUTLASS 2.x    : " << cutlass_avg_ms << " ms  |  "
              << std::setprecision(2) << cutlass_tflops << " TFLOPS\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  cuBLAS         : " << cublas_avg_ms << " ms  |  "
              << std::setprecision(2) << cublas_tflops << " TFLOPS\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    if (cutlass_avg_ms > 0 && cublas_avg_ms > 0) {
        std::cout << "  Speedup (CUTLASS / cuBLAS) : "
                  << cublas_avg_ms / cutlass_avg_ms << "x\n";
    }
    std::cout << "=============================================================\n";

    // 释放资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cutlass));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return 0;
}
