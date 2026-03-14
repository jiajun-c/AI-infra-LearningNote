#include <iostream>
#include <cuda_runtime.h>  // 只需包含这个，无需 <cuda.h>
#include <assert.h>
using namespace std;

__global__ void Kernel(int *mem) {  // 修正拼写
    __shared__ int x;
    void* p = &x;
    uint32_t smem32 = __cvta_generic_to_shared(p);

    size_t smem64 = smem32;
    void* q = __cvta_shared_to_generic(smem64);
    assert(q == p);
}

int main() {
    int *dGlobal;
    cudaMalloc((void**)&dGlobal, sizeof(int));
    
    Kernel<<<1,1>>>(dGlobal);
    cudaDeviceSynchronize();  // 等待 kernel 完成
    
    cudaFree(dGlobal);  // 释放内存（良好习惯）
    return 0;
}