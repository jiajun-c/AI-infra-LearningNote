#include <cuda.h>
#include <cuda/barrier>
#include <iostream>
using namespace std;
# export CUDA_VISIBLE_DEVICES=3
static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int *data, size_t offset) {
    __shared__ alignas(16) int smem_data[buf_len];
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cuda::device::experimental::fence_proxy_async_shared_cta();// b)
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        cuda::memcpy_async(smem_data, 
                           data + offset,
                           cuda::aligned_size_t<16>(sizeof(smem_data)),
                         bar)
    }
}

int main() {

}