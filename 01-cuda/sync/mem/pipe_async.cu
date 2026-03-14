#include <cuda.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <iostream>
#define CHECK(call)                                \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)
using namespace std;
template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count, int batch) {
  extern __shared__ T s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();
  for (int t = 0; t < batch; t++) {

    for (size_t i = 0; i < copy_count; ++i) {
        shared[threadIdx.x] = global[threadIdx.x];
    }
  }
  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count, int batch) {
  extern __shared__ T s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();
  for (int t = 0; t < batch; t++) {
  //pipeline pipe;
    for (size_t i = 0; i < copy_count; ++i) {
        __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                                &global[blockDim.x * i + threadIdx.x], sizeof(T));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }
  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

int main() {
    cudaSetDevice(3);
    int block_size = 256;
    int grid_size = 1;
    int copy_count = block_size * grid_size;
    double *d_global;
    cudaMalloc(&d_global, copy_count * sizeof(double));
    uint64_t*  d_clock_sync, *d_clock_async;
    cudaMalloc(&d_clock_sync, sizeof(uint64_t));
    cudaMemset(d_clock_sync, 0, sizeof(uint64_t));

    cudaMalloc(&d_clock_async, sizeof(uint64_t));
    cudaMemset(d_clock_async, 0, sizeof(uint64_t));
    CHECK(cudaGetLastError());
    for (int i = 0; i < 10; i++) {
        pipeline_kernel_async<double><<<grid_size, block_size, copy_count * sizeof(double)>>>(
            d_global, d_clock_async, 1, 4);
        pipeline_kernel_sync<double><<<grid_size, block_size, copy_count * sizeof(double)>>>(
            d_global, d_clock_sync, 1, 4);
    }
    cudaDeviceSynchronize();
    cudaMemset(d_clock_async, 0, sizeof(uint64_t));
    cudaMemset(d_clock_sync, 0, sizeof(uint64_t));
     pipeline_kernel_sync<double><<<grid_size, block_size, copy_count * sizeof(double)>>>(
            d_global, d_clock_sync, 1, 4);
        pipeline_kernel_async<double><<<grid_size, block_size, copy_count * sizeof(double)>>>(
            d_global, d_clock_async, 1, 4);
    CHECK(cudaGetLastError());
    uint64_t *h_clock_sync, *h_clock_async;
    cudaDeviceSynchronize();

    h_clock_sync = (uint64_t*)malloc(sizeof(uint64_t));
    h_clock_async = (uint64_t*)malloc(sizeof(uint64_t));
    cudaMemcpy(h_clock_sync, d_clock_sync, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clock_async, d_clock_async, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    printf("h_clock_sync clock: %d\n", *h_clock_sync);
    printf("h_clock_async clock: %d\n", *h_clock_async);

}