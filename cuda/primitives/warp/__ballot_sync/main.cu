#include <cuda.h>
#include <iostream>
#define N 1000
__global__ void ballot(int *a, int *b, int n)
 {
     int tid = threadIdx.x;
     if (tid > n)
         return;
     b[tid] = __ballot_sync(0xffffffff, a[tid] > 12);
 }

int main() {
    int *a = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) a[i] = i%24;
    int *b = (int *)malloc(N * sizeof(int));
    int* dev_a;
    int* dev_b;
    cudaMalloc(&dev_a, N*sizeof(int));
    cudaMalloc(&dev_b, N*sizeof(int));

    cudaMemcpy(dev_a, a,  N*sizeof(int), cudaMemcpyHostToDevice);
    ballot<<<1, N>>>(dev_a, dev_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%d ", b[i]);
    }
    free(a);
    free(b);
    cudaFree(dev_a);
    cudaFree(dev_b);
}