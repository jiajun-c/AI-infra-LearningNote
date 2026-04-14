#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvtx3/nvToolsExt.h>

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

    // 2. 定义矩阵维度
    const int M = 16384;
    const int K = 16384;
    const int N = 1;

    size_t size_A = M * K * sizeof(float);
    size_t size_x = K * N * sizeof(float);
    size_t size_y = M * N * sizeof(float);

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

    // 创建矩阵布局
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, M));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, M));

    // 分配 Workspace
    size_t workspaceSize = 4 * 1024 * 1024;
    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    float alpha = 1.0f, beta = 0.0f;

    // 5. 测试不同 SM 限制
    std::vector<int> sm_targets = {8, 4, 2, 1};

    std::cout << std::left << std::setw(15) << "Target SMs"
              << std::setw(15) << "Time (ms)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int sm_count : sm_targets) {
        cublasLtMatmulDesc_t matmulDesc;
        CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        // 设置 SM 数量目标
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            matmulDesc,
            CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
            &sm_count,
            sizeof(sm_count)
        ));

        // 获取启发式算法
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

        // 预热
        for (int i = 0; i < 5; i++) {
            CHECK_CUBLASLT(cublasLtMatmul(
                handle, matmulDesc, &alpha,
                d_A, layoutA, d_x, layoutB, &beta,
                d_y, layoutC, d_y, layoutC,
                &heuristicResult.algo, d_workspace, workspaceSize, 0
            ));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // NVTX Range - 方便用 nsys 分析
        std::string rangeName = "SM_" + std::to_string(sm_count);
        nvtxRangePushA(rangeName.c_str());

        // 计时
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

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

        std::cout << std::left << std::setw(15) << sm_count
                  << std::setw(15) << (ms / num_iters) << std::endl;

        nvtxRangePop();

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUBLASLT(cublasLtMatmulDescDestroy(matmulDesc));
    }

    // 清理
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
