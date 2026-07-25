#include <algorithm>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#define WARP_SIZE 32
using namespace std;

// 256
// 8 warp
template<const int warpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = warpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


__global__ void block_all_reduce_sum_f32_f32(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    __shared__ float reduce_smem[8];
    int warpID = tid / WARP_SIZE;
    int laneID = tid % WARP_SIZE;
    // 32 * 256 
    float sum = (idx < N) ? a[idx] : 0.0f;
    sum = warp_reduce_sum_f32(sum);
    if (laneID == 0) {
        reduce_smem[warpID] = sum;
    }
    __syncthreads();
    sum = (laneID < 8) ? reduce_smem[laneID]: 0.0f;
    if (warpID == 0) sum = warp_reduce_sum_f32(sum);
    if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32x4_f32(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid)*4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1)/ WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float4 reg_a = *reinterpret_cast<float4*>(&a[idx]);
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.w + reg_a.z) : 0.0f;
    sum = warp_reduce_sum_f32(sum);
    int laneID = threadIdx.x % WARP_SIZE;
    int warpID = threadIdx.x / WARP_SIZE;
    if (laneID == 0) {
        reduce_smem[warpID] = sum;
    }
    __syncthreads();
    sum = (laneID < NUM_WARPS) ? reduce_smem[laneID] : 0.0f;

    if (warpID == 0) {
        sum = warp_reduce_sum_f32(sum);
    }

    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

template<const int KWarpSize = WARP_SIZE>
__device__ __forceinline__ half 
warp_reduce_sum_fp8_e4m3_fp16(__nv_fp8_storage_t val) {
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
    #pragma unroll
    for (int mask = KWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    }
    return val_f16;
}

template<const int KWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp16_fp16(half val) {
    #pragma unroll
    for (int mask = KWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_fp8_e4m3_fp16_kernel(__nv_fp8_storage_t *a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];

    __nv_fp8_storage_t sum_f8 = (idx < N) ? a[idx] : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
    
    int warpID = tid / WARP_SIZE;
    int laneID = tid % WARP_SIZE;
    half sum_f16 = warp_reduce_sum_fp8_e4m3_fp16(sum_f8);
    if (laneID == 0) reduce_smem[warpID] = sum_f16;
    __syncthreads();

    half sum = (laneID < NUM_WARPS) ? reduce_smem[laneID]: __float2half(0.0f);
    if (warpID == 0) {
        sum = warp_reduce_sum_fp16_fp16(sum);
    }
    if (tid == 0) {
        atomicAdd(y, __half2float(sum));
    }
}

int main() {
    // 1. 设置测试规模和并行参数
    int N = 1024 * 1024 * 128; // 128M 个 FP8 元素 (约 128MB)
    constexpr int NUM_THREADS = 256;
    int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

    // 2. 主机端分配内存与初始化
    // __nv_fp8_e4m3 是 Host 和 Device 都可以操作的 FP8 类型
    __nv_fp8_e4m3* h_a = new __nv_fp8_e4m3[N];
    float h_y = 0.0f;

    // 简单初始化：将所有元素设为 1.0 (在 E4M3 下是可以精确表示的)
    for (int i = 0; i < N; ++i) {
        h_a[i] = __nv_fp8_e4m3(1.0f); 
    }

    // 3. 设备端分配显存
    // Kernel 签名使用的是 __nv_fp8_storage_t (底层其实是 uint8_t)
    __nv_fp8_storage_t* d_a;
    float* d_y;
    cudaMalloc((void**)&d_a, N * sizeof(__nv_fp8_storage_t));
    cudaMalloc((void**)&d_y, sizeof(float));

    // 4. 数据拷贝与状态重置 (关键步)
    cudaMemcpy(d_a, h_a, N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    
    // 必须将接收 atomicAdd 的全局地址清零，否则会包含显存里的随机垃圾值
    cudaMemset(d_y, 0, sizeof(float)); 

    // 5. 发起 Kernel 调用
    // __nv_fp8_e4m3 的内存布局与 __nv_fp8_storage_t 一致，直接传入即可
    block_all_reduce_fp8_e4m3_fp16_kernel<NUM_THREADS><<<num_blocks, NUM_THREADS>>>(
        d_a, 
        d_y, 
        N
    );

    // 6. 同步并检查错误
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 7. 取回结果
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Expected Sum: " << (float)N << std::endl;
    std::cout << "Actual Sum:   " << h_y << std::endl;

    // 8. 释放资源
    cudaFree(d_a);
    cudaFree(d_y);
    delete[] h_a;

    return 0;
}