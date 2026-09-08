// 双调排序测试代码（单 block 版本，对应 sort.cu）
// 编译：nvcc test_sort.cu sort.cu -o test_sort
// 运行：./test_sort

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

// ==================== 待测试的 kernel（修正笔误 kk → k）====================
__global__ void bitonic_sort(int* values, int len) {
    for (int k = 2; k <= len; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
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
            __syncthreads();   // 每步比较交换后必须同步，否则下一次比较看到旧值
        }
    }
}

// ==================== 工具：CUDA 错误检查 ====================
#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(e));              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ==================== 单一 case 的测试 ====================
// return true 表示通过
bool run_one_case(int len, int trials, std::mt19937& rng) {
    if ((len & (len - 1)) != 0) {
        fprintf(stderr, "[skip] len=%d 不是 2 的幂\n", len);
        return false;
    }

    std::vector<int> h_data(len);
    std::vector<int> h_ref(len);

    // 单 block 最多 1024 线程；这里我们让线程数 = len
    int threads = len;
    if (threads > 1024) {
        fprintf(stderr, "[skip] len=%d 超过单 block 线程上限 1024\n", len);
        return false;
    }

    bool all_pass = true;

    for (int t = 0; t < trials; ++t) {
        // 生成 [-1000, 1000] 之间的随机整数（含负数）
        std::uniform_int_distribution<int> dist(-1000, 1000);
        for (int i = 0; i < len; ++i) {
            h_data[i] = dist(rng);
            h_ref[i]  = h_data[i];
        }

        // GPU 端 buffer
        int* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, len * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                              len * sizeof(int),
                              cudaMemcpyHostToDevice));

        // 单 block，threads 个线程
        bitonic_sort<<<1, threads>>>(d_data, len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 拷回
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                              len * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // 用 std::sort 做参考
        std::sort(h_ref.begin(), h_ref.end());

        // 比对
        bool ok = std::equal(h_data.begin(), h_data.end(), h_ref.begin());
        if (!ok) {
            printf("  [FAIL] len=%4d trial=%d\n", len, t);
            printf("    GPU : ");
            for (int i = 0; i < len && i < 16; ++i) printf("%d ", h_data[i]);
            printf("...\n");
            printf("    REF : ");
            for (int i = 0; i < len && i < 16; ++i) printf("%d ", h_ref[i]);
            printf("...\n");
            all_pass = false;
        }

        cudaFree(d_data);
    }

    if (all_pass) {
        printf("  [PASS] len=%4d  trials=%d\n", len, trials);
    }
    return all_pass;
}

// ==================== main ====================
int main() {
    std::mt19937 rng(42);   // 固定种子，结果可复现

    const int trials_per_len = 30;

    // 测试 2 的各次幂，从 2 到 1024
    int lens[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    printf("Bitonic Sort test (single block, len must be power of 2)\n");
    printf("Threads per block = len, trials per len = %d\n\n", trials_per_len);

    int total_cases = 0, pass_cases = 0;
    for (int len : lens) {
        ++total_cases;
        if (run_one_case(len, trials_per_len, rng)) ++pass_cases;
    }

    printf("\n==== %d / %d cases passed ====\n", pass_cases, total_cases);
    return (pass_cases == total_cases) ? 0 : 1;
}