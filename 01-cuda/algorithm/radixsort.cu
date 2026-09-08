#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
const int N = 1024;

__global__ void radixsort(unsigned int* v) {
    __shared__ unsigned int key[N];
    __shared__ unsigned int scan[N];
    const int t = threadIdx.x;
    key[t] = v[t];
    __syncthreads();
    for (int bit = 0; bit < 32; bit++) {
        const unsigned int now = key[t];
        const unsigned int b = (x >> bit) & 1u;
        scan[t] = 1 - b;
        __syncthreads();
        for (int d = 1; d < N; d <<= 1) {
            int add = (t >= d) ? scan[t - d]: 0u;
            __syncthreads();
            scan[t] += add;
            __syncthreads();
        }
        int totalZeros = scan[N-1];
        int zeroBefore = scan[t] - (1u - b);
        int onesBefore = t - zeroBefore;
        int dst = (b == 0u) ? zeroBefore: totalZeros + onesBefore;
        __syncthreads();
        key[dst] = now;
        __syncthreads();
    }
    v[t] = key[t]; 
}