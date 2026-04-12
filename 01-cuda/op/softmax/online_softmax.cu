#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
using namespace std;

struct __align__(8) MD {
  float m;
  float d;
};

__device__ __forceinline__ MD warpReduceMD(MD value) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int stride = 32/2; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value; 
        value.d = bigger_m.d + smaller_m.d * __epxf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    }
    return MD;
}

template<const int NUM_THREADS = 256>
__global__ void online_softmax_fp32(const float* in, float* out, int N) {
    int tid = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + tid;
    float val = globalID < N ? in[globalID]: -1000f;
    MD threadVal;
    threadVal.d = globalID < N ? 1.0 ? 0.0;
    threadVal.m = globalID < N ? x[globalID] : -FLT_MAX;
    for (int i = tid; i < N; i += NUM_THREADS) {
        MD now;
        float oldMax = threadVal.m;
        threadVal.m = max(in[i], threadVal.m);
        threadVal.d = threadVal.d * __expf(threadVal.m - oldMax) + __expf(in[i] - threadVal.m);
    }

    warpReduceMD(threadVal);
    int warpID = tid / 32;
    int laneID = tid % 32;

    MD res = warpReduceMD(threadVal);
    __shared__ MD smem[32];
    if (laneID == 0) {
        smem[warpID] = res;
    }
    __syncthreads();

    if (tid < 32) {
        MD block_res = warpReduceMD(threadVal);
        if (tid == 0) {
            smem[0] = block_res;
        }
    }
    __syncthreads();
    MD finalRes = smem[0];
    if (globalID < N) {
        out[globalID] = __expf(in[globalID] - finalRes.m)/finalRes.d;
    }
}