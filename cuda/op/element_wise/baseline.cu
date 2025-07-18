#include <cuda.h>
#include <cuda_runtime.h>

__global__ void baseline_kernel(float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx] < 0 ? 0 : input[idx];
  }
}

int main() {
    float* input;
    float* output;
    int32_t element_num = 100* 1024* 1024 ;
    cudaMalloc(&input, element_num * sizeof(float));
    cudaMalloc(&output, element_num * sizeof(float));
    int32_t thread_num = 256;
    int32_t grid_size = (element_num + thread_num -1) / thread_num;
    baseline_kernel<<<grid_size, thread_num>>>(input, output, element_num);
    cudaDeviceSynchronize();
    cudaFree(input);
    cudaFree(output);
    return 0;
  }