#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>

__global__ void baseline_kernel(half *input,half *input1, half *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx] * input1[idx];
  }
}

int main() {
    half* input, *input1;
    half* output;
    int32_t element_num = 32* 1024* 1024 ;
    cudaMalloc(&input, element_num * sizeof(half));
    cudaMalloc(&input1, element_num * sizeof(half));
    cudaMalloc(&output, element_num * sizeof(half));
    int32_t thread_num = 256;
    int32_t grid_size = (element_num + thread_num -1) / thread_num;
    baseline_kernel<<<grid_size, thread_num>>>(input, input1, output, element_num);
    cudaDeviceSynchronize();
    cudaFree(input);
    cudaFree(output);
    return 0;
  }