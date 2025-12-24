#include <iostream>
#include <cuda_runtime.h>  // 只需包含这个，无需 <cuda.h>

using namespace std;

__global__ void Kernel(int *mem) {  // 修正拼写
    __shared__ int smem[100];
    printf("mem: isGlobal: %d isSem: %d \n", __isGlobal(mem), __isShared(mem));
    printf("smem: isGlobal: %d isSem: %d \n", __isGlobal(smem), __isShared(smem));

}

int main() {
    int *dGlobal;
    cudaMalloc((void**)&dGlobal, sizeof(int));
    
    Kernel<<<1,1>>>(dGlobal);
    cudaDeviceSynchronize();  // 等待 kernel 完成
    
    cudaFree(dGlobal);  // 释放内存（良好习惯）
    return 0;
}