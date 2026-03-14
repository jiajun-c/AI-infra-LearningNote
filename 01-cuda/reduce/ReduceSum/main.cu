#include <cuda.h>
#include <iostream>
#define N 128

// 在warp层面进行reduceSum的操作
template <typename T>
__inline__  __device__ int WarpReduceSum(T val) {
    for (int offset = (warpSize >> 1); offset > 0; offset >>=1) {
        val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
    }
    return val;
}

template<typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
    int laneid = threadIdx.x % warpSize;
    int warpid = threadIdx.x / warpSize;
    val = WarpReduceSum(val);
    __syncthreads();
    if (laneid == 0) {
        shared[warpid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : T(0);
    if (warpid == 0) {
      val = WarpReduceSum(val);
    }
    return val;
}

template<typename T>
__global__ void blockReduceKernel(const T* input, T* output, int n) {
    extern __shared__ T shared[];  // 动态分配共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T val = (idx < n) ? input[idx] : 0.0f;  // 读取输入值
    // printf("val in: %d\n", val);
    val = BlockReduceSum(val, shared);          // 调用 BlockReduceSum

    if (tid == 0) {
        output[blockIdx.x] = val;  // 将结果写入输出数组
    }
}

int main() {
    int *a = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) a[i] = 1;
    int *b = (int *)malloc(N * sizeof(int));
    int* dev_a;
    int* dev_b;
    cudaMalloc(&dev_a, N*sizeof(int));
    cudaMalloc(&dev_b, N*sizeof(int));

    cudaMemcpy(dev_a, a,  N*sizeof(int), cudaMemcpyHostToDevice);
    blockReduceKernel<<<2, N/2, N * sizeof(float)>>>(dev_a, dev_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("sum %d %d\n", b[0], b[1]);
    free(a);
    free(b);
    cudaFree(dev_a);
    cudaFree(dev_b);
}