#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublasLt.h>

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLASLT(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt Error at " << __FILE__ << ":" << __LINE__ \
                      << " - code " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 简单的内核用于初始化数据
__global__ void initData(float* data, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

int main() {
    // 1. 获取当前 GPU 设备信息
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    int total_sms = prop.multiProcessorCount;
    std::cout << "Device: " << prop.name << " | Total SMs: " << total_sms << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 2. 定义矩阵维度 (GEMV: A(M, K) * x(K, 1) = y(M, 1))
    // 设得足够大，以充分体现带宽压力
    const int M = 16384; 
    const int K = 16384;
    const int N = 1;     // N=1 意味着这是 GEMV
    
    size_t size_A = M * K * sizeof(float);
    size_t size_x = K * N * sizeof(float);
    size_t size_y = M * N * sizeof(float);

    // 计算一次 GEMV 读取和写入的总字节数 (A + x + y)
    double total_bytes = (double)(size_A + size_x + size_y);

    // 3. 分配显存
    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_x, size_x));
    CHECK_CUDA(cudaMalloc(&d_y, size_y));

    // 初始化数据
    int threads = 256;
    initData<<<(M * K + threads - 1) / threads, threads>>>(d_A, M * K, 1.0f);
    initData<<<(K + threads - 1) / threads, threads>>>(d_x, K, 1.0f);
    initData<<<(M + threads - 1) / threads, threads>>>(d_y, M, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 初始化 cuBLASLt
    cublasLtHandle_t handle;
    CHECK_CUBLASLT(cublasLtCreate(&handle));

    // 创建矩阵布局 (注意 cuBLAS 默认是列主序 Column-Major)
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, M));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, M));

    // 分配 Workspace (GEMV 通常不需要很大 workspace，给 4MB 足矣)
    size_t workspaceSize = 4 * 1024 * 1024;
    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 5. 定义要测试的 SM 数量列表 (0 表示使用全部硬件 SM)
    std::vector<int> sm_targets = {0, total_sms, total_sms/2, total_sms/4, total_sms/8, 8, 4, 2, 1};

    std::cout << std::left << std::setw(15) << "Target SMs" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(20) << "Bandwidth (GB/s)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int sm_count : sm_targets) {
        // 创建操作描述符
        cublasLtMatmulDesc_t matmulDesc;
        CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        // 🔥 核心控制：设置 SM 数量目标 (CUDA 11.8+ / cuBLAS 11.11.3+)
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            matmulDesc, 
            CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, 
            &sm_count, 
            sizeof(sm_count)
        ));

        // 根据当前的 SM 限制，查询最合适的启发式算法 (Heuristic)
        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        int returnedResults = 0;
        CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
            handle, matmulDesc, layoutA, layoutB, layoutC, layoutC, 
            pref, 1, &heuristicResult, &returnedResults
        ));

        if (returnedResults == 0) {
            std::cerr << "No algorithm found for SM count: " << sm_count << std::endl;
            cublasLtMatmulDescDestroy(matmulDesc);
            continue;
        }

        // 预热 (Warm-up)
        for (int i = 0; i < 5; i++) {
            CHECK_CUBLASLT(cublasLtMatmul(
                handle, matmulDesc, &alpha, 
                d_A, layoutA, d_x, layoutB, &beta, 
                d_y, layoutC, d_y, layoutC, 
                &heuristicResult.algo, d_workspace, workspaceSize, 0
            ));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // 计时测试
        const int num_iters = 20;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < num_iters; i++) {
            CHECK_CUBLASLT(cublasLtMatmul(
                handle, matmulDesc, &alpha, 
                d_A, layoutA, d_x, layoutB, &beta, 
                d_y, layoutC, d_y, layoutC, 
                &heuristicResult.algo, d_workspace, workspaceSize, 0
            ));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iters;
        
        // 计算显存带宽 (GB/s)
        double bandwidth = (total_bytes / 1e9) / (avg_ms / 1000.0);

        std::string sm_label = (sm_count == 0) ? "0 (All)" : std::to_string(sm_count);
        std::cout << std::left << std::setw(15) << sm_label
                  << std::setw(15) << avg_ms 
                  << std::setw(20) << bandwidth << std::endl;

        // 销毁描述符
        CHECK_CUBLASLT(cublasLtMatmulDescDestroy(matmulDesc));
    }

    // 6. 清理资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(layoutC));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(pref));
    CHECK_CUBLASLT(cublasLtDestroy(handle));

    return 0;
}