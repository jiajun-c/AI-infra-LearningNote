#include <cuda.h>
#include <cuda_runtime.h>

__gloabal__ void prefixSum(float *data, float* out) {
    __shared__ float x[N];

    int t = threadIdx.x;
    x[t] = data[t];
    __syncthreads();
    for (int d = 1; d < N; d <<= 1) {
        float value = (t >= d) ? x[t - d]: 0;
        __syncthreads();
        x[t] += value;
    }
    out[t] = x[t];
}

int main() {
    
}