#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cuda_fp16.h>
using namespace std;
#define WARP_SIZE 32
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))


#define div_ceil(a, b) (((a) + (b) - 1) / (b))
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))


#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))


__global__ void mmaNaiveKernel(const half * __restrict__ A, const half * __restrict__ B, half * __restrict__ C, size_t M, size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;
    if (warp_row >= M || warp_col >= N) return;
    __shared__ half A_shared[MMA_M][MMA_K];
    __shared__ half B_shared[MMA_N][MMA_K];
    __shared__ half C_shared[MMA_M][MMA_N];
    const size_t lane_id = threadIdx.x % 32;
    uint32_t RC[2] = {0, 0};
    #pragma unroll
    for (size_t k_tile = 0; k_tile < K_tiles; k_tile++) { 
        *((int4*)(&A_shared[lane_id/2][0]) + lane_id%2) =  *((int4 *)(&A[(warp_row + lane_id / 2) * K + k_tile * MMA_K]) + lane_id % 2);
        if (lane_id < MMA_K) {
            for (size_t i = 0; i < MMA_N; i++) {
                B_shared[i][lane_id] = B[(warp_col + i) + (k_tile * MMA_K + lane_id)*N ];
            }
        }
        __syncthreads();
        uint32_t RA[4];
        uint32_t RB[2];
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_shared[lane_id/16][(lane_id/16)*8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_shared[lane_id/8][((lane_id/8)%2)*8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }
    *((uint32_t *)(&C_shared[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_shared[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];
    if (lane_id < MMA_M) {
        *((int4*)(&C[(warp_row + lane_id)*N + warp_col])) = *((int4*)(&C_shared[lane_id][0]));
    }
}

void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    half *h_C = (half*)malloc(M * N * sizeof(half));
    for (int i = 0; i < M * K; i++) {   
        h_A[i] = 1;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1;
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0;
    }
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(half), cudaMemcpyHostToDevice);

    for (int i = 0; i < 1; i++) {
        auto start = chrono::high_resolution_clock::now();
        mmaNaive(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        // cout << "Time taken by function: " << duration.count() ;
    }
    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    printf("C[1][1]: %f\n", float(h_C[0]));
}