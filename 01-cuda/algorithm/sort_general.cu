// 支持任意长度的双调排序
// 编译：nvcc sort_general.cu -o sort_general
// 用法：./sort_general

#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

// ==================== 工具 ====================
inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(e));              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ==================== Kernels ====================

// 阶段 kernel：处理一个 (k, j) 阶段
// 多个 block，每个 thread 处理一个 idx
__global__ void bitonic_step(int* values, int padded_len, int k, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_len) return;

    int p = idx ^ j;
    if (p > idx) {
        bool keepmin = ((idx & k) == 0);
        int a = values[idx], b = values[p];
        if ((a > b) == keepmin) {
            values[idx] = b;
            values[p]   = a;
        }
    }
}

// 把 [start, start+n) 填充为 INT_MAX（哨兵值，永远排在末尾）
__global__ void fill_int_max(int* values, int start, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) values[start + idx] = INT_MAX;
}

// ==================== Host 调度 ====================
// 输入：
//   d_values : 长度为 padded_len 的 device buffer（>= len）
//   len      : 真实数据长度（任意正整数）
//   padded_len : len 向上取整到的下一个 2 的幂，由 next_pow2(len) 计算
//
// 步骤：
//   1) 把 [len, padded_len) 填充为 INT_MAX
//   2) 对 padded_len 范围内的元素跑完整双调排序
//   3) 完成后真实数据 [0, len) 已有序；尾部 padding 元素均为 INT_MAX
void bitonic_sort_general(int* d_values, int len) {
    if (len <= 0) return;
    int padded_len = next_pow2(len);

    const int threads = 256;

    // 1) padding
    if (padded_len > len) {
        int pad = padded_len - len;
        int blocks = (pad + threads - 1) / threads;
        fill_int_max<<<blocks, threads>>>(d_values, len, pad);
    }

    // 2) 多阶段排序
    for (int k = 2; k <= padded_len; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int blocks = (padded_len + threads - 1) / threads;
            bitonic_step<<<blocks, threads>>>(d_values, padded_len, k, j);
            // 这里不需要 cudaDeviceSynchronize：
            // 同一个 stream 的下一次 kernel launch 会隐式等待这一次完成
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== 测试 ====================
bool run_one_case(int len, int trials, std::mt19937& rng) {
    int padded_len = next_pow2(len);

    std::vector<int> h_data(len);
    std::vector<int> h_ref(len);

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, padded_len * sizeof(int)));

    bool all_pass = true;
    std::uniform_int_distribution<int> dist(-10000, 10000);

    for (int t = 0; t < trials; ++t) {
        for (int i = 0; i < len; ++i) {
            h_data[i] = dist(rng);
            h_ref[i]  = h_data[i];
        }

        // 把 d_data 的全部 padded_len 个 int 拷入（多余部分会被 padding 覆盖）
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                              len * sizeof(int),
                              cudaMemcpyHostToDevice));

        bitonic_sort_general(d_data, len);

        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                              len * sizeof(int),
                              cudaMemcpyDeviceToHost));

        std::sort(h_ref.begin(), h_ref.end());

        if (!std::equal(h_data.begin(), h_data.end(), h_ref.begin())) {
            printf("  [FAIL] len=%5d padded=%5d trial=%d\n",
                   len, padded_len, t);
            all_pass = false;
        }
    }

    if (all_pass) {
        printf("  [PASS] len=%5d padded=%5d  trials=%d\n",
               len, padded_len, trials);
    }
    cudaFree(d_data);
    return all_pass;
}

int main() {
    std::mt19937 rng(2026);

    // 覆盖各种长度：1、小非2幂、中等非2幂、大数、超百万
    int lens[] = {
        1, 3, 7, 13, 31, 100, 511, 1000,
        2048, 5000, 65535, 100000, 1 << 20, 1 << 24
    };

    int trials = 10;       // 大数组少跑几次，控制耗时
    int total = 0, passed = 0;

    printf("Bitonic Sort (any length, padded to next pow2)\n\n");
    for (int len : lens) {
        ++total;
        if (len > (1 << 22)) trials = 2;       // 太大时只跑 2 次
        else if (len > (1 << 18)) trials = 3;
        else trials = 10;
        if (run_one_case(len, trials, rng)) ++passed;
    }

    printf("\n==== %d / %d cases passed ====\n", passed, total);
    return (passed == total) ? 0 : 1;
}