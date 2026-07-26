#include <cstdio>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void test_conflict_cycles(float *out_latency, bool use_swizzle)
{
    // 声明两种 Shared Memory：一种有冲突，一种无冲突 (Padding)
    __shared__ float tile_bad[32][32];
    __shared__ float tile_good[32][33];

    // 初始化一下，防止编译器优化掉
    int tid = threadIdx.x;
    int row = tid; 
    int col = 0; // 所有线程读第0列 -> 制造冲突

    if (use_swizzle) {
        tile_good[row][col] = (float)tid;
    } else {
        tile_bad[row][col] = (float)tid;
    }
    __syncthreads();

    // --- 开始计时 ---
    long long start_clock = clock64();

    float val;
    if (use_swizzle) {
        // 无冲突访问: tile_good[row][col] 
        // 这里的 row = tid, col = 0。由于 Padding，它们在不同的 Bank。
        val = tile_good[row][0]; 
    } else {
        // 有冲突访问: tile_bad[row][col]
        // 所有线程访问 tile_bad[tid][0]。所有地址模 32 都等于 0 -> Bank 0。
        val = tile_bad[row][0];
    }

    // 制造数据依赖，防止编译器把读操作优化掉
    // 只有把 val 写出去，编译器才认为读操作是必须的
    if (val > 10000.0f) out_latency[0] = val; 

    // --- 结束计时 ---
    long long end_clock = clock64();

    // 只让第0号线程把结果写回去 (代表整个 Warp 的经历)
    if (tid == 0) {
        out_latency[use_swizzle ? 1 : 0] = (float)(end_clock - start_clock);
    }
}

int main() {
    float *d_latency;
    float h_latency[2] = {0};
    cudaMalloc(&d_latency, 2 * sizeof(float));

    // 只启动 1 个 Block，1 个 Warp (32 threads)
    // 这样能最大限度排除线程调度带来的噪声
    printf("Running latency test...\n");
    
    // 测试 1: 有冲突 (False)
    test_conflict_cycles<<<1, 32>>>(d_latency, false);
    
    // 测试 2: 无冲突 (True)
    test_conflict_cycles<<<1, 32>>>(d_latency, true);

    cudaMemcpy(h_latency, d_latency, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Naive (Conflict) Cycles: %.0f\n", h_latency[0]);
    printf("Swizzle (No Conflict) Cycles: %.0f\n", h_latency[1]);
    printf("Ratio (Bad / Good): %.2fx\n", h_latency[0] / h_latency[1]);

    cudaFree(d_latency);
    return 0;
}