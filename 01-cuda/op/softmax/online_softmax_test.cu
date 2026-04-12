#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// ============================================================================
// 数据结构
// ============================================================================

struct __align__(8) MD {
    float m;  // max
    float d;  // sum of exp
};

// ============================================================================
// Device Functions
// ============================================================================

__device__ __forceinline__ MD warpReduceMD(MD value) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int stride = 16; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        if (value.m > other.m) {
            value.d = value.d + other.d * __expf(other.m - value.m);
        } else {
            value.d = value.d * __expf(value.m - other.m) + other.d;
            value.m = other.m;
        }
    }
    return value;
}

// 单块 softmax
template<const int NUM_THREADS = 256>
__global__ void online_softmax_fp32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N
) {
    int tid = threadIdx.x;

    // 初始化：不使用 -FLT_MAX 作为初始值，避免数值问题
    MD threadVal;
    threadVal.m = -FLT_MAX;
    threadVal.d = 0.0f;

    for (int i = tid; i < N; i += NUM_THREADS) {
        float x = in[i];
        if (threadVal.m == -FLT_MAX) {
            threadVal.m = x;
            threadVal.d = 1.0f;
        } else {
            float oldMax = threadVal.m;
            threadVal.m = fmaxf(x, threadVal.m);
            threadVal.d = threadVal.d * __expf(threadVal.m - oldMax) + __expf(x - threadVal.m);
        }
    }

    MD warpVal = warpReduceMD(threadVal);

    int laneID = tid % 32;
    int warpID = tid / 32;
    int numWarps = (NUM_THREADS + 31) / 32;

    extern __shared__ char smem_raw[];
    MD* smem = reinterpret_cast<MD*>(smem_raw);

    if (laneID == 0) {
        smem[warpID] = warpVal;
    }
    __syncthreads();

    // block 级别 reduce - 同样需要处理初始值
    MD blockVal;
    blockVal.m = -FLT_MAX;
    blockVal.d = 0.0f;

    if (tid < 32) {
        for (int i = tid; i < numWarps; i += 32) {
            if (blockVal.m == -FLT_MAX) {
                blockVal = smem[i];
            } else {
                float oldMax = blockVal.m;
                blockVal.m = fmaxf(smem[i].m, blockVal.m);
                blockVal.d = blockVal.d * __expf(blockVal.m - oldMax) + smem[i].d * __expf(smem[i].m - blockVal.m);
            }
        }
        blockVal = warpReduceMD(blockVal);
        if (tid == 0) {
            smem[0] = blockVal;
        }
    }
    __syncthreads();

    MD finalRes = smem[0];
    if (tid < N) {
        out[tid] = __expf(in[tid] - finalRes.m) / finalRes.d;
    }
}

// 多块 softmax - 每个 block 处理一行
__global__ void online_softmax_fp32_multi_block(
    const float* __restrict__ in,
    float* __restrict__ out,
    int batch_size,
    int N
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    const float* row_input = in + row * N;
    float* row_output = out + row * N;

    MD threadVal;
    threadVal.m = -FLT_MAX;
    threadVal.d = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = row_input[i];
        if (threadVal.m == -FLT_MAX) {
            threadVal.m = x;
            threadVal.d = 1.0f;
        } else {
            float oldMax = threadVal.m;
            threadVal.m = fmaxf(x, threadVal.m);
            threadVal.d = threadVal.d * __expf(threadVal.m - oldMax) + __expf(x - threadVal.m);
        }
    }

    MD warpVal = warpReduceMD(threadVal);

    int laneID = tid % 32;
    int warpID = tid / 32;
    int numWarps = (blockDim.x + 31) / 32;

    extern __shared__ char smem_raw[];
    MD* smem = reinterpret_cast<MD*>(smem_raw);

    if (laneID == 0) {
        smem[warpID] = warpVal;
    }
    __syncthreads();

    MD blockVal;
    blockVal.m = -FLT_MAX;
    blockVal.d = 0.0f;

    if (tid < 32) {
        for (int i = tid; i < numWarps; i += 32) {
            if (blockVal.m == -FLT_MAX) {
                blockVal = smem[i];
            } else {
                float oldMax = blockVal.m;
                blockVal.m = fmaxf(smem[i].m, blockVal.m);
                blockVal.d = blockVal.d * __expf(blockVal.m - oldMax) + smem[i].d * __expf(smem[i].m - blockVal.m);
            }
        }
        blockVal = warpReduceMD(blockVal);
        if (tid == 0) {
            smem[0] = blockVal;
        }
    }
    __syncthreads();

    MD finalRes = smem[0];
    if (tid < N) {
        row_output[tid] = __expf(row_input[tid] - finalRes.m) / finalRes.d;
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

void softmax_cpu_reference(const float* input, float* output, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= sum_exp;
    }
}

// ============================================================================
// 工具函数
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
}

// ============================================================================
// 测试函数
// ============================================================================

bool test_softmax_correctness(int N, float tolerance = 1e-5) {
    const int threads = 256;
    // 根据 N 计算需要的 block 数
    const int blocks = (N + threads - 1) / threads;

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    std::vector<float> h_output_ref(N);

    std::srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int numWarps = (threads + 31) / 32;
    size_t smem_size = numWarps * sizeof(MD);

    // 使用 multi-block kernel
    online_softmax_fp32_multi_block<<<blocks, threads, smem_size>>>(d_input, d_output, 1, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    softmax_cpu_reference(h_input.data(), h_output_ref.data(), N);

    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_output[i] - h_output_ref[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > tolerance) passed = false;
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    float sum_error = fabsf(sum - 1.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "N=" << std::setw(5) << N << ": max_diff=" << std::setw(12) << max_diff
              << ", sum_error=" << std::setw(12) << sum_error
              << " [" << (passed && sum_error < tolerance ? "PASSED" : "FAILED") << "]" << std::endl;

    return passed && (sum_error < tolerance);
}

bool test_softmax_batched(int batch_size, int N) {
    const int threads = 256;

    std::vector<float> h_input(batch_size * N);
    std::vector<float> h_output(batch_size * N);
    std::vector<float> h_output_ref(batch_size * N);

    std::srand(42);
    for (int i = 0; i < batch_size * N; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * N * sizeof(float), cudaMemcpyHostToDevice));

    int numWarps = (threads + 31) / 32;
    size_t smem_size = numWarps * sizeof(MD);

    online_softmax_fp32_multi_block<<<batch_size, threads, smem_size>>>(d_input, d_output, batch_size, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;
    float max_diff = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        softmax_cpu_reference(h_input.data() + b * N, h_output_ref.data() + b * N, N);
        for (int i = 0; i < N; i++) {
            float diff = fabsf(h_output[b * N + i] - h_output_ref[b * N + i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff > 1e-4f) passed = false;
        }
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Batch=" << batch_size << ", N=" << N << ": max_diff=" << max_diff
              << " [" << (passed ? "PASSED" : "FAILED") << "]" << std::endl;

    return passed;
}

void benchmark_softmax(int N, int num_iters = 100) {
    const int threads = 256;

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int numWarps = (threads + 31) / 32;
    size_t smem_size = numWarps * sizeof(MD);

    online_softmax_fp32<threads><<<1, threads, smem_size>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        online_softmax_fp32<threads><<<1, threads, smem_size>>>(d_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float total_bytes = 2 * N * sizeof(float);
    float avg_time_ms = elapsed_ms / num_iters;
    float bandwidth_gbs = (total_bytes / avg_time_ms) / 1e6;

    std::cout << "N=" << std::setw(6) << N << ": time=" << std::setw(10) << avg_time_ms
              << "ms, bandwidth=" << std::setw(10) << bandwidth_gbs << " GB/s" << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_device_info();
    std::cout << std::endl;

    std::cout << "=== Correctness Test (Single Block) ===" << std::endl;
    std::vector<int> test_sizes = {32, 64, 128, 256, 512, 1024};
    bool all_passed = true;
    for (int N : test_sizes) {
        if (!test_softmax_correctness(N)) {
            all_passed = false;
        }
    }
    std::cout << std::endl;

    std::cout << "=== Batched Test (Multi Block) ===" << std::endl;
    std::vector<std::pair<int, int>> batch_configs = {
        {32, 128}, {64, 256}, {128, 512}, {256, 1024}
    };
    for (auto [bs, N] : batch_configs) {
        if (!test_softmax_batched(bs, N)) {
            all_passed = false;
        }
    }
    std::cout << std::endl;

    std::cout << "=== Benchmark ===" << std::endl;
    std::vector<int> bench_sizes = {256, 512, 1024, 2048, 4096, 8192};
    for (int N : bench_sizes) {
        benchmark_softmax(N);
    }
    std::cout << std::endl;

    std::cout << "Overall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_passed ? 0 : 1;
}
