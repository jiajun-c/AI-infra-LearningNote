// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: warp1 naive hgemv

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK
#define COLS_PRE_WARP 2
#define COLS_PRE_BLOCK 8
#define THREADS_PRE_GROUP 16

__global__ void warp2NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
                                 size_t K) {
    // 以warp为单位进行执行，每个warp处理一行
    const size_t group_id = threadIdx.x / THREADS_PRE_GROUP;
    const size_t group_col = blockIdx.x * COLS_PRE_BLOCK + group_id;
    if (group_col >= N) return;
    const size_t K_iter = div_ceil(K, COLS_PRE_WARP);
    const size_t group_lane_id = threadIdx.x % THREADS_PRE_GROUP;
    float tmp = 0;
    #pragma unroll
    for (size_t i = 0; i < K_iter; ++i) {
        size_t A_idx = i * THREADS_PRE_GROUP + group_lane_id;
        size_t B_idx = i * THREADS_PRE_GROUP + group_lane_id + group_col*K;
        tmp += __half2float(A[A_idx]) * __half2float(B[B_idx]);
    }
    constexpr unsigned int mask = 0xffffffff;
    
    #pragma unroll
    for (size_t i = THREADS_PRE_GROUP / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (group_lane_id == 0) {
        C[group_col] = __float2half(tmp);
    }
}
void warp2Naive(half *A, half *B, half *C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, WARPS_PER_BLOCK));

    warp2NaiveKernel<<<grid, block>>>(A, B, C, N, K);
}