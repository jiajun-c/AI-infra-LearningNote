#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
using namespace std;

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
__device__ __forceinline__ void mma_m8n8k4(double *acc, double &frag_a, double &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

__global__ void warpReduceMulSum(double *a, double *b, double* c, int m, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x & 31;
    int warpId = tid >> 5;
    if (warpId >= m) return;
    double sum = 0;
    double fragA = 1.0, fragB = 1.0;
    for (int i = 0; i < n/32; i++) {
        // fragA = a[warpId * n + i * 32 + laneId];
        // fragB = b[warpId * n + i * 32 + laneId];
        sum += fragA * fragB;
    }
    for (int offset = 32/2; offset > 0; offset /= 2) 
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (laneId == 0) 
        c[warpId] = sum;
}

__global__ void mmaReduceSum(double *a, double *b, double* c, int m, int n) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x & 31;
    int warpId = tid >> 5;
    // double fragA = a[warpId * n + laneId];
    // double fragB = b[warpId * n + laneId];
    double fragA = 1.0, fragB = 1.0, fragC[2] = {0};
    for (int i = 0; i < n/32; i++) {
        // fragA = a[warpId * n + i * 32 + laneId];
        // fragB = b[warpId * n + i * 32 + laneId];
        mma_m8n8k4(fragC, fragA, fragB);
        // fragC[0] += fragA + fragB;
    }
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 9, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 9, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    // fragC[0] += __shfl_sync(0xffffffff, fragC[1], 4);

    if (laneId == 0) 
        c[warpId] = fragC[0];

}

int main() { 
    cudaSetDevice(1);
    int m = 1024*8;
    int n = 2048;
    int warmup = 10;
    int repeat = 100;
    double *d_a, *d_b, *d_c;
    // cudaMalloc(&d_a, m * n * sizeof(double));
    // cudaMalloc(&d_b, m * n * sizeof(double));
    cudaMalloc(&d_c, m * sizeof(double));
    cudaMemset(d_c, 0, m * sizeof(double));
    for (int i = 0; i < warmup ; i++) {
        warpReduceMulSum<<<m, 32>>>(d_a, d_b, d_c, m, n);
    }
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
        warpReduceMulSum<<<m, 32>>>(d_a, d_b, d_c, m, n);
        cudaDeviceSynchronize();
    }
    auto end =  chrono::high_resolution_clock::now();
    auto time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "Average execution time of warpReduceMulSum kernel: " << time * 1e-3f / repeat << " ms" << endl;

    for (int i = 0; i < warmup ; i++) {
        mmaReduceSum<<<m, 32>>>(d_a, d_b, d_c, m, n);
    }
    CHECK(cudaGetLastError());
    start = chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
        mmaReduceSum<<<m, 32>>>(d_a, d_b, d_c, m, n);
        cudaDeviceSynchronize();
    }

    end =  chrono::high_resolution_clock::now();
    time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "Average execution time of mmaReduceSum kernel: " << time * 1e-3f / repeat << " ms" << endl;
}