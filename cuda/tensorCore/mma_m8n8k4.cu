#include <iostream>

using namespace std;
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

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


__global__ void mma_fp64_acc_fp64(double *out) { 
    double acc[2]= {0};
    // acc[0] = 0.0;
    // acc[0] = 0.0;
    int tid = threadIdx.x;
    if (tid >= 32) return;
    double frag_a = double(tid);
    double frag_b =0.0;
    if (tid%4 == 0) frag_b = double(tid/4);
    mma_m8n8k4(acc, frag_a, frag_b);
    out[tid*2] = acc[0];
    out[tid*2+1] = acc[1];
}
int main()
{
    double *out;
    cudaMalloc(&out, 64*sizeof(double));
    mma_fp64_acc_fp64<<<1, 32>>>(out); 
    (cudaDeviceSynchronize());

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(kernelErr));
    }
    double *out_cpu;
    out_cpu = (double *)malloc(64*sizeof(double));
    cudaMemcpy(out_cpu, out, 64*sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            cout << out_cpu[i*8+j] << " ";
        }
        printf("\n");
    }
}