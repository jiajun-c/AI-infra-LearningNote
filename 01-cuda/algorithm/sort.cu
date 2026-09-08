#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void sort(int* values, int len) {
    for (int k = 2; k <= len; k <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
                int p = idx ^ j;
                if (p > idx) {
                    bool keepmin = ((idx & k) == 0);
                    int a = values[idx], b = values[p];
                    if ((a > b) == keepmin) {
                        values[idx] = b;
                        values[p]   = a;
                    }
                }
            }
            __syncthreads();
        }
    } 
}

