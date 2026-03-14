#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void sequence_gpu(int *d_ptr, int N) {
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemID < N) {
        unsigned int laneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        d_ptr[elemID] = laneid;
    }
}


int main() {
    int *d_ptr, N = 1024;
    cudaMalloc(&d_ptr, N * sizeof(int));

    int *h_ptr;
    cudaMallocHost(&h_ptr, N * sizeof(int));
    
    sequence_gpu<<<N / 256, 256>>>(d_ptr, N);

}