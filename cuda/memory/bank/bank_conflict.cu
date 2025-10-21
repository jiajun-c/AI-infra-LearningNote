#include <cuda.h>
#include <iostream>

__global__ void transSmem(float* out) {
    __shared__ float matrix[32][32];
    matrix[threadIdx.y][threadIdx.x ^ threadIdx.y] = out[threadIdx.x + 32*threadIdx.y]; // 加载原始数据
    __syncthreads();
    out[threadIdx.x + 32*threadIdx.y] = matrix[threadIdx.x][threadIdx.x ^ threadIdx.y]; // 写入转置数据
}

int main() {
    dim3 block{32, 32};
    float *d_out, *out;
    cudaMalloc((void**)&d_out, 32*32*4);
    out = (float*)malloc(sizeof(float)*32*32);
    for (int i = 0; i < 32*32; i++) {
        out[i] = i;
    }
    cudaMemcpy(d_out, out, 32*32*sizeof(float), cudaMemcpyHostToDevice);
    transSmem<<<1, block>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, 32*32*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32;i++) {
        for (int j = 0; j < 32; j++) {
            printf("%.2f ", out[i*32+j]);
        }
        printf("\n");
    }

}