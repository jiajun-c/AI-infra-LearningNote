#include <iostream>
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ void mma_spm16n8k16(uint32_t  *acc, uint32_t * frag_a, uint32_t * frag_b, int metadata)
{
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3}, {%4,%5}, {%6,%7}, %8, 0x0;\n"
        : "=r"(acc[0]), "=r"(acc[1])//, "=f"(D[2]), "=f"(D[3])
        : "r"(frag_a[0]), "r"(frag_a[1]), 
          "r"(frag_b[0]), "r"(frag_b[1]),
          "r"(acc[0]), "r"(acc[1]), //"f"(C[2]), "f"(C[3]),
          "r"(metadata)
    );
}

__global__ void matmul(half *A, half *B, half *C) { 
    half fraga[4], fragb[4];
    fraga[0] = __int2half_rd(2);
    fraga[1] = __int2half_rd(1);
    fraga[2] = __int2half_rd(1);
    fraga[3] = __int2half_rd(1);
    fragb[0] = __int2half_rd(0);
    fragb[1] = __int2half_rd(1);
    fragb[2] = __int2half_rd(0);
    fragb[3] = __int2half_rd(1);
    half acc[4] = {0};
    uint32_t* fraga_int = reinterpret_cast<uint32_t*>(&fraga);
    uint32_t* fragb_int = reinterpret_cast<uint32_t*>(&fragb);
    uint32_t* acc_int = reinterpret_cast<uint32_t*>(&acc);
    int metadata = 0x77777777;
    if (threadIdx.x%4 != 0) metadata = 0;
    if (threadIdx.x/4 == 0) metadata = 0;
    mma_spm16n8k16(acc_int, fraga_int, fragb_int, metadata);
    C[(threadIdx.x*2)] = acc[0];
    C[(threadIdx.x*2)+1] = acc[1];
    C[(threadIdx.x*2) + 64] = acc[2];
    C[(threadIdx.x*2)+ 64+1] = acc[3];
}

int main() {
    half *A, *B, *C;
    A = (half *)malloc(sizeof(half) * 16*16);
    B = (half *)malloc(sizeof(half) * 16*8);
    C = (half *)malloc(sizeof(half) * 16*8);

    half *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(half) * 16*16);
    cudaMalloc((void **)&d_B, sizeof(half) * 16*8);
    cudaMalloc((void **)&d_C, sizeof(half) * 16*8);
    matmul<<<1, 32>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, sizeof(half) * 16*8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << float(C[i*8+j]) << " ";
        }
        std::cout << std::endl;
    }
}