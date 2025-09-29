#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#define WARP_SIZE 32
#define WARPS_PRE_BLOCK 4
#define THREADS_PRE_BLOCK 128
#define div_ceil(x, y) ((x + y - 1) / y) 

__global__ void threadNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
    size_t N, size_t K) {
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= N) return;
        float temp = 0.0f;
        #pragma unroll
        for (size_t i = 0; i < K; i++) {
            temp += __half2float(A[i * N + col]) * __half2float(B[i]);
        }
        C[col] = __float2half(temp);
    }

void threadNaive(half* A, half* B, half* C, size_t N, size_t K) {
    dim3 block(THREADS_PRE_BLOCK);
    dim3 grid(div_ceil(N, THREADS_PRE_BLOCK));
    threadNaiveKernel<<<grid, block>>>(A, B, C, N, K);
}
