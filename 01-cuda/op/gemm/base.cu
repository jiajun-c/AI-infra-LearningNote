#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>

template<const int TILE_SIZE = 16>
__global__ void gemm_base_NT(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 计算当前线程负责计算 C 中的哪个全局元素的坐标
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    // 只需要 A 和 B 的 Shared Memory
    __shared__ float smemA[TILE_SIZE][TILE_SIZE];
    __shared__ float smemB[TILE_SIZE][TILE_SIZE];

    // 🔥 性能优化：使用寄存器来存放累加结果，不仅清零了，还极速
    float sum = 0.0f;

    // 2. 沿着 K 维度进行分块推进
    for (int i = 0; i < K; i += TILE_SIZE) {
        
        // 载入 A 的 Tile (注意边界检查)
        int write_col_A = tidx ^ tidy;
        if (row < M && (i + tidy) < K) {
            smemA[tidy][write_col_A] = A[row * K + i + tidx];
        } else {
            smemA[tidy][write_col_A] = 0.0f; // 越界填 0
        }

        // 载入 B 的 Tile 
        // 因为是 NT (B 是转置的，B 的形状也是 N x K，所以行索引是 col，列索引是 i)
        if (col < N && (i + tidx) < K) {
            smemB[tidy][write_col_A] = B[col * K + i + tidx];
        } else {
            smemB[tidy][write_col_A] = 0.0f;
        }

        // 同步，等待所有线程将当前 Tile 的数据搬运完毕
        __syncthreads();

        // 3. 计算当前 Tile 的乘加
        for (int j = 0; j < TILE_SIZE; j++) {
            // 注意这里：对于 NT 布局，B 的读取逻辑与标准的 NN 布局不同。
            // 因为 B 是转置存储的，A 的第 j 列要和 B 的第 j 列相乘
            sum += smemA[write_col_A][j] * smemB[tidy][j]; 
        }

        // 🔥 极其关键：防止跑得快的线程进入下一个循环覆盖 smem 数据
        __syncthreads();
    }

    // 4. 将最终结果写回 C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}