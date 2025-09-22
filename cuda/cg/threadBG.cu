#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
namespace cg = cooperative_groups;

__global__ void kernel(int *input) {
    __shared__ int x;
    cg::thread_block tb = cg::this_thread_block();
    printf("thread_rank %d\n", tb.thread_rank());
    if (tb.thread_rank() == 0) x = (*input);
    tb.sync();
}

int main() {
    int *d_input;
    cudaMalloc(&d_input, sizeof(int));
    kernel<<<2, 32>>>(d_input);
    cudaDeviceSynchronize();

}
