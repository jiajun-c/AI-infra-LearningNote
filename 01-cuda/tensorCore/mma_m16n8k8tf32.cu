#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>

using namespace std;
__device__ __forceinline__
void tf32_m16n8k8(float* MatA, float* MatB, float* MatC) {
    int const* A   = reinterpret_cast<int const*>(MatA);
    int const* B   = reinterpret_cast<int const*>(MatB);
    float* C        = reinterpret_cast<float*>(MatC);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
        :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
         "r"(B[0]), "r"(B[1])
    );
}

__global__ void mma_ftf32_acc_tf32(float* matA, float*matB, float *out) { 
    int tid = threadIdx.x;
    float fragA[4], fragB[2], fragC[4] = {0.0};
    // for (int i = 0; i < 4; i++) fragA[i] = 1.0, fragC[i] = 0.0;
    fragA[0] = matA[tid%4 + tid/4*8];
    fragA[1] = matA[tid%4 + tid/4*8 + 64];
    fragA[2] = matA[tid%4 + tid/4*8 + 4];
    fragA[3] = matA[tid%4 + tid/4*8 + 68];
    fragB[0] = matB[tid/4 + tid%4*8];
    fragB[1] = matB[tid/4 + tid%4*8+32];
    // for (int i = 0; i < 2; i++) fragB[i] = 1.0;
    tf32_m16n8k8(fragA, fragB, fragC);
    // out[tid*2] = fragC[0];
    // out[tid*2+1] = fragC[1];

    out[tid*2] = fragC[0];
    out[tid*2+1] = fragC[1];
    out[tid*2+64] = fragC[2];
    out[tid*2+64+1] = fragC[3];
}
int main() {
    int m = 16;
    int n = 8;
    int k = 8;
    float *hMatA = (float*)malloc(m * k * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            hMatA[i * k + j] = i;
        }
    } 

    float *hMatB = (float*)malloc(k * n * sizeof(float));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            hMatB[i * n + j] = i;
        }
    }
    float *hMatC = (float*)malloc(m * n * sizeof(float));

    float *dMatA, *dMatB, *dMatC;

    cudaMalloc((void**)&dMatA, m * k * sizeof(float));
    cudaMalloc((void**)&dMatB, n * k * sizeof(float));
    cudaMalloc((void**)&dMatC, m * n * sizeof(float));
    cudaMemcpy(dMatA, hMatA, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMatB, hMatB, n * k * sizeof(float), cudaMemcpyHostToDevice);
    mma_ftf32_acc_tf32<<<1,32>>>(dMatA, dMatB, dMatC);
    cudaMemcpy(hMatC, dMatC, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << hMatC[i * n + j] << " ";
        }
        cout << endl;
    }
}

