#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// ---------------------------------------------------------
// Kernel 1: SMID-based Multi-GEMV
// 不同 SM 组计算不同的输出向量
// 使用单独的参数传递每个向量的指针，避免指针数组
// ---------------------------------------------------------
__global__ void multi_gemv_smid(const float* __restrict__ A,
                                const float* __restrict__ X0, const float* __restrict__ X1,
                                const float* __restrict__ X2, const float* __restrict__ X3,
                                float* __restrict__ Y0, float* __restrict__ Y1,
                                float* __restrict__ Y2, float* __restrict__ Y3,
                                int N, int M, int num_vectors,
                                int sm_count) {
    int laneID = threadIdx.x % 32;
    int local_warpID = threadIdx.x / 32;
    unsigned int my_smid = get_smid();

    if (my_smid >= sm_count) return;

    // 每个 SM 负责一个特定的输出向量
    int warps_per_block = blockDim.x / 32;
    int global_warpID = my_smid * warps_per_block + local_warpID;

    // SM 分组：每 (132 / num_vectors) 个 SM 负责一个向量
    int sms_per_vector = (sm_count + num_vectors - 1) / num_vectors;
    int vector_id = my_smid / sms_per_vector;
    if (vector_id >= num_vectors) return;

    // 在该向量内部，使用全局 warp ID 分配行
    int warps_for_this_vector = sms_per_vector * warps_per_block;
    int local_warp_in_vector = global_warpID % warps_for_this_vector;

    // 根据 vector_id 选择指针
    const float* X = (vector_id == 0) ? X0 : (vector_id == 1) ? X1 :
                     (vector_id == 2) ? X2 : X3;
    float* Y = (vector_id == 0) ? Y0 : (vector_id == 1) ? Y1 :
               (vector_id == 2) ? Y2 : Y3;

    for (int row = local_warp_in_vector; row < N; row += warps_for_this_vector) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) Y[row] = partial_sum;
    }
}

// ---------------------------------------------------------
// Kernel 2: WarpID-based Multi-GEMV (对照组)
// ---------------------------------------------------------
__global__ void multi_gemv_warpid(const float* __restrict__ A,
                                  const float* __restrict__ X0, const float* __restrict__ X1,
                                  const float* __restrict__ X2, const float* __restrict__ X3,
                                  float* __restrict__ Y0, float* __restrict__ Y1,
                                  float* __restrict__ Y2, float* __restrict__ Y3,
                                  int N, int M, int num_vectors,
                                  int active_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = tid / 32;
    int laneID = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int max_active_warps = active_blocks * warps_per_block;

    if (warpID >= max_active_warps) return;

    // Warp 分组
    int warps_per_vector = (max_active_warps + num_vectors - 1) / num_vectors;
    int vector_id = warpID / warps_per_vector;
    if (vector_id >= num_vectors) return;

    int local_warp_in_vector = warpID % warps_per_vector;

    const float* X = (vector_id == 0) ? X0 : (vector_id == 1) ? X1 :
                     (vector_id == 2) ? X2 : X3;
    float* Y = (vector_id == 0) ? Y0 : (vector_id == 1) ? Y1 :
               (vector_id == 2) ? Y2 : Y3;

    for (int row = local_warp_in_vector; row < N; row += warps_per_vector) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) Y[row] = partial_sum;
    }
}

// ---------------------------------------------------------
// Kernel 3: Standard Multi-GEMV (对照组)
// ---------------------------------------------------------
__global__ void multi_gemv_standard(const float* __restrict__ A,
                                    const float* __restrict__ X0, const float* __restrict__ X1,
                                    const float* __restrict__ X2, const float* __restrict__ X3,
                                    float* __restrict__ Y0, float* __restrict__ Y1,
                                    float* __restrict__ Y2, float* __restrict__ Y3,
                                    int N, int M, int num_vectors) {
    int rows_per_block = (N + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, N);
    int laneID = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    // 每个 block 负责一个向量
    int vector_id = blockIdx.x % num_vectors;

    const float* X = (vector_id == 0) ? X0 : (vector_id == 1) ? X1 :
                     (vector_id == 2) ? X2 : X3;
    float* Y = (vector_id == 0) ? Y0 : (vector_id == 1) ? Y1 :
               (vector_id == 2) ? Y2 : Y3;

    for (int row = row_start + warpID; row < row_end; row += warps_per_block) {
        float partial_sum = 0.0f;
        for (int col = laneID; col < M; col += 32) {
            partial_sum += A[row * M + col] * X[col];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (laneID == 0) Y[row] = partial_sum;
    }
}

// ---------------------------------------------------------
// Kernel 4: Simple Multi-GEMV
// ---------------------------------------------------------
__global__ void multi_gemv_simple(const float* __restrict__ A,
                                  const float* __restrict__ X0, const float* __restrict__ X1,
                                  const float* __restrict__ X2, const float* __restrict__ X3,
                                  float* __restrict__ Y0, float* __restrict__ Y1,
                                  float* __restrict__ Y2, float* __restrict__ Y3,
                                  int N, int M, int num_vectors) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneID = threadIdx.x % 32;

    // 每个 thread 负责一个 (vector_id, row) 对
    int total_rows = N * num_vectors;
    if (global_tid >= total_rows) return;

    int vector_id = global_tid / N;
    int row = global_tid % N;

    const float* X = (vector_id == 0) ? X0 : (vector_id == 1) ? X1 :
                     (vector_id == 2) ? X2 : X3;
    float* Y = (vector_id == 0) ? Y0 : (vector_id == 1) ? Y1 :
               (vector_id == 2) ? Y2 : Y3;

    float sum = 0.0f;
    for (int col = laneID; col < M; col += 32) {
        sum += A[row * M + col] * X[col];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (laneID == 0) Y[row] = sum;
}

// ---------------------------------------------------------
// CPU 参考实现
// ---------------------------------------------------------
void gemv_cpu_reference(const float* A, const float* X, float* Y, int N, int M) {
    for (int i = 0; i < N; i++) {
        Y[i] = 0.0f;
        for (int j = 0; j < M; j++) {
            Y[i] += A[i * M + j] * X[j];
        }
    }
}

bool verify_results(const float* expected, const float* actual, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < N; i++) {
        float diff = std::abs(expected[i] - actual[i]);
        float rel_error = diff / (std::abs(expected[i]) + 1e-8f);
        if (rel_error > tolerance && diff > tolerance) {
            std::cerr << "验证失败：位置 " << i << ", 期望=" << expected[i]
                      << ", 实际=" << actual[i] << ", 相对误差=" << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

enum KernelType { SMID, WARPID, STANDARD, SIMPLE, CUBLAS };

float test_multi_gemv(KernelType type, const float* d_A,
                      const float* d_X0, const float* d_X1, const float* d_X2, const float* d_X3,
                      float* d_Y0, float* d_Y1, float* d_Y2, float* d_Y3,
                      int N, int M, int num_vectors,
                      int blocks_limit, int threads_per_block,
                      cublasHandle_t cublas_handle) {
    int launch_blocks = 132;

    if (type == CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        const float* X_ptrs[] = {d_X0, d_X1, d_X2, d_X3};
        float* Y_ptrs[] = {d_Y0, d_Y1, d_Y2, d_Y3};
        for (int v = 0; v < num_vectors; v++) {
            CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T,
                                      M, N,
                                      &alpha,
                                      d_A, M,
                                      X_ptrs[v], 1,
                                      &beta,
                                      Y_ptrs[v], 1));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            for (int v = 0; v < num_vectors; v++) {
                cublasSgemv(cublas_handle, CUBLAS_OP_T, M, N,
                            &alpha, d_A, M, X_ptrs[v], 1, &beta, Y_ptrs[v], 1);
            }
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / iterations;

    } else {
        if (type == SMID) {
            multi_gemv_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors, blocks_limit);
        } else if (type == WARPID) {
            multi_gemv_warpid<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors, blocks_limit);
        } else if (type == STANDARD) {
            multi_gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors);
        } else {
            multi_gemv_simple<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            if (type == SMID) {
                multi_gemv_smid<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors, blocks_limit);
            } else if (type == WARPID) {
                multi_gemv_warpid<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors, blocks_limit);
            } else if (type == STANDARD) {
                multi_gemv_standard<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors);
            } else {
                multi_gemv_simple<<<launch_blocks, threads_per_block>>>(d_A, d_X0, d_X1, d_X2, d_X3, d_Y0, d_Y1, d_Y2, d_Y3, N, M, num_vectors);
            }
        }
        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / iterations;
    }
}

int main() {
    std::vector<std::pair<int, int>> test_sizes = {
        {1024, 128},
        {1024, 256},
        {1024, 512},
        {2048, 128},
        {2048, 256},
        {2048, 512},
        {4096, 128},
        {4096, 256},
        {8192, 128}
    };

    std::vector<int> num_vectors_list = {1, 2, 4};
    int threads_per_block = 256;

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    for (auto& [N, M] : test_sizes) {
        size_t size_A = N * M * sizeof(float);
        size_t size_X = M * sizeof(float);
        size_t size_Y = N * sizeof(float);

        std::vector<float> h_A(N * M, 1.0f);
        std::vector<std::vector<float>> h_X_vecs;
        for (int v = 0; v < 4; v++) {
            std::vector<float> h_X(M, 1.0f + v * 0.1f);
            h_X_vecs.push_back(h_X);
        }

        float *d_A;
        std::vector<float*> d_X_vecs(4);
        std::vector<float*> d_Y_vecs(4);

        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        for (int v = 0; v < 4; v++) {
            CHECK_CUDA(cudaMalloc(&d_X_vecs[v], size_X));
            CHECK_CUDA(cudaMalloc(&d_Y_vecs[v], size_Y));
            CHECK_CUDA(cudaMemcpy(d_X_vecs[v], h_X_vecs[v].data(), size_X, cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));

        std::cout << "\n========================================================" << std::endl;
        std::cout << "测试矩阵大小：" << N << " x " << M << " (FP32)" << std::endl;
        std::cout << "========================================================\n" << std::endl;

        std::cout << std::left << std::setw(12) << "Vectors"
                  << "  " << std::setw(14) << "SMID"
                  << "  " << std::setw(14) << "WarpID"
                  << "  " << std::setw(14) << "Standard"
                  << "  " << std::setw(14) << "Simple"
                  << "  " << std::setw(14) << "cuBLAS"
                  << "  " << std::setw(12) << "SMID 优势" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------" << std::endl;

        for (int num_vectors : num_vectors_list) {
            float ms_smid = test_multi_gemv(SMID, d_A,
                                            d_X_vecs[0], d_X_vecs[1], d_X_vecs[2], d_X_vecs[3],
                                            d_Y_vecs[0], d_Y_vecs[1], d_Y_vecs[2], d_Y_vecs[3],
                                            N, M, num_vectors, 132, threads_per_block, cublas_handle);
            float ms_warpid = test_multi_gemv(WARPID, d_A,
                                              d_X_vecs[0], d_X_vecs[1], d_X_vecs[2], d_X_vecs[3],
                                              d_Y_vecs[0], d_Y_vecs[1], d_Y_vecs[2], d_Y_vecs[3],
                                              N, M, num_vectors, 132, threads_per_block, cublas_handle);
            float ms_standard = test_multi_gemv(STANDARD, d_A,
                                                d_X_vecs[0], d_X_vecs[1], d_X_vecs[2], d_X_vecs[3],
                                                d_Y_vecs[0], d_Y_vecs[1], d_Y_vecs[2], d_Y_vecs[3],
                                                N, M, num_vectors, 132, threads_per_block, cublas_handle);
            float ms_simple = test_multi_gemv(SIMPLE, d_A,
                                              d_X_vecs[0], d_X_vecs[1], d_X_vecs[2], d_X_vecs[3],
                                              d_Y_vecs[0], d_Y_vecs[1], d_Y_vecs[2], d_Y_vecs[3],
                                              N, M, num_vectors, 132, threads_per_block, cublas_handle);
            float ms_cublas = test_multi_gemv(CUBLAS, d_A,
                                              d_X_vecs[0], d_X_vecs[1], d_X_vecs[2], d_X_vecs[3],
                                              d_Y_vecs[0], d_Y_vecs[1], d_Y_vecs[2], d_Y_vecs[3],
                                              N, M, num_vectors, 132, threads_per_block, cublas_handle);

            double bytes = num_vectors * ((double)N * M * 4.0 + M * 4.0 + N * 4.0);
            double bw_smid = (bytes / (ms_smid / 1000.0)) / 1e9;
            double bw_warpid = (bytes / (ms_warpid / 1000.0)) / 1e9;
            double bw_standard = (bytes / (ms_standard / 1000.0)) / 1e9;
            double bw_simple = (bytes / (ms_simple / 1000.0)) / 1e9;
            double bw_cublas = (bytes / (ms_cublas / 1000.0)) / 1e9;

            float advantage = (bw_smid - bw_warpid) / bw_warpid * 100;

            std::cout << std::left << std::setw(12) << num_vectors
                      << "  " << std::setw(14) << (std::to_string((int)(bw_smid * 100) / 100.0) + " GB/s")
                      << "  " << std::setw(14) << (std::to_string((int)(bw_warpid * 100) / 100.0) + " GB/s")
                      << "  " << std::setw(14) << (std::to_string((int)(bw_standard * 100) / 100.0) + " GB/s")
                      << "  " << std::setw(14) << (std::to_string((int)(bw_simple * 100) / 100.0) + " GB/s")
                      << "  " << std::setw(14) << (std::to_string((int)(bw_cublas * 100) / 100.0) + " GB/s")
                      << "  " << std::setw(12) << (std::to_string((int)(advantage * 100) / 100.0) + "%")
                      << std::endl;
        }

        CHECK_CUDA(cudaFree(d_A));
        for (int v = 0; v < 4; v++) {
            CHECK_CUDA(cudaFree(d_X_vecs[v]));
            CHECK_CUDA(cudaFree(d_Y_vecs[v]));
        }
    }

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    return 0;
}
