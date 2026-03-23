#include <iostream>
#include <cuda_runtime.h>

__device__ float warpReduceMax(float val) {
    for (int i = 16; i >= 1; i >>= 1) {
        val = fmax(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

// fix 1024 threads
__device__ float blockReduceMax(float val) {
    __shared__ float smem[32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    float maxVal = warpReduceMax(val);
    if (laneID == 0) smem[warpID] = maxVal;
    __syncthreads();
    maxVal = warpReduceMax(maxVal);
    return maxVal;
}

__global__ void reduceMax(float* in, float* out, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float maxVal = idx < N? in[idx]: -1e9;
    for (int i = tid; i < N; i += blockDim.x) {
        maxVal = max(maxVal, in[blockDim.x * blockIdx.x + i]);
    }
    maxVal = blockReduceMax(maxVal);
    if (tid == 0) {
        out[blockIdx.x] = maxVal;
    }
}