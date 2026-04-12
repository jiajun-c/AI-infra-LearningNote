#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

using namespace std;

struct __align__(8) MD {
    float m;
    float d;
};

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

template<const int NUM_THREADS = 256>
__global__ void online_softmax_fp32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    MD* debug_smem  // 用于调试
) {
    int tid = threadIdx.x;

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
        if (debug_smem) debug_smem[warpID] = warpVal;  // 调试输出
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
        out[tid] = __expf(in[tid] - finalRes.m) / finalRes.d;
    }
}

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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    const int N = 64;
    const int threads = 256;
    int numWarps = (threads + 31) / 32;

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    std::vector<float> h_cpu_output(N);
    std::vector<MD> h_smem(numWarps);

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i * 0.1f - 3.0f;  // 简单的递增序列
    }

    float *d_input, *d_output;
    MD *d_smem;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_smem, numWarps * sizeof(MD)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    size_t smem_size = numWarps * sizeof(MD);

    online_softmax_fp32<threads><<<1, threads, smem_size>>>(d_input, d_output, N, d_smem);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_smem.data(), d_smem, numWarps * sizeof(MD), cudaMemcpyDeviceToHost));

    std::cout << "Shared memory values:" << std::endl;
    for (int i = 0; i < numWarps; i++) {
        std::cout << "  warp[" << i << "]: m=" << h_smem[i].m << ", d=" << h_smem[i].d << std::endl;
    }

    softmax_cpu_reference(h_input.data(), h_cpu_output.data(), N);

    std::cout << "\nInput: ";
    for (int i = 0; i < 5; i++) std::cout << h_input[i] << " ";
    std::cout << "..." << std::endl;

    std::cout << "GPU Output: ";
    for (int i = 0; i < 5; i++) std::cout << h_output[i] << " ";
    std::cout << "..." << std::endl;

    std::cout << "CPU Output: ";
    for (int i = 0; i < 5; i++) std::cout << h_cpu_output[i] << " ";
    std::cout << "..." << std::endl;

    float gpu_sum = 0, cpu_sum = 0;
    for (int i = 0; i < N; i++) {
        gpu_sum += h_output[i];
        cpu_sum += h_cpu_output[i];
    }
    std::cout << "\nGPU sum: " << gpu_sum << ", CPU sum: " << cpu_sum << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_smem));

    return 0;
}
