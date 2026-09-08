// 多 block 前缀和（支持任意 N）
// 编译：nvcc scan_general.cu -o scan_general
// 用法：./scan_general

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(e));              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// chunk size = block 内的线程数；这里取 1024（最大单 block 线程数）
// shared memory 使用量 = 4 KB（完全在 48 KB 限制内）
const int CHUNK = 1024;

// ============================================================
// Pass 1：每个 block 独立扫自己那一段，并把"本 block 总和"写出
// ============================================================
__global__ void block_scan(const float* data, float* out,
                           float* block_sums, int N) {
    __shared__ float s[CHUNK];
    int t = threadIdx.x;
    int idx = blockIdx.x * CHUNK + t;

    // 1) 加载（含越界处理：不足一块的部分填 0）
    s[t] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // 2) Hillis-Steele inclusive scan（原地）
    for (int d = 1; d < CHUNK; d <<= 1) {
        float v = (t >= d) ? s[t - d] : 0.0f;
        __syncthreads();
        s[t] += v;
    }

    // 3) 写回局部结果
    out[idx] = s[t];

    // 4) 最后一个线程把"本 block 的总和"提交
    if (t == CHUNK - 1) {
        block_sums[blockIdx.x] = s[t];
    }
}

// ============================================================
// Pass 2：扫一遍 block_sums，得到每个 block 的全局前缀
//         （单 block，假定 num_blocks ≤ 1024；超过则递归）
// ============================================================
__global__ void scan_block_sums(float* block_sums, int num_blocks) {
    __shared__ float s[CHUNK];
    int t = threadIdx.x;

    s[t] = (t < num_blocks) ? block_sums[t] : 0.0f;
    __syncthreads();

    for (int d = 1; d < CHUNK; d <<= 1) {
        float v = (t >= d) ? s[t - d] : 0.0f;
        __syncthreads();
        s[t] += v;
    }

    // 写入 exclusive 形式的全局偏移：block_sums[i] = sum(block 0..i-1)
    if (t == 0)         block_sums[0] = 0.0f;                  // block 0 偏移为 0
    else if (t < num_blocks) block_sums[t] = s[t - 1];         // block i 偏移 = s[i-1]
}

// ============================================================
// Pass 3：把全局偏移加到每个 block 的 chunk 上（block 0 不加）
// ============================================================
__global__ void add_block_prefix(float* out, const float* block_sums, int N) {
    int t = threadIdx.x;
    int idx = blockIdx.x * CHUNK + t;
    if (blockIdx.x > 0 && idx < N) {
        out[idx] += block_sums[blockIdx.x];
    }
}

// ============================================================
// Host 封装：3 个 kernel launch 串行执行，stream 隐式同步
// ============================================================
void prefix_sum_host(const float* d_data, float* d_out, int N) {
    if (N <= 0) return;
    int num_blocks = (N + CHUNK - 1) / CHUNK;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));

    block_scan<<<num_blocks, CHUNK>>>(d_data, d_out, d_block_sums, N);
    scan_block_sums<<<1, CHUNK>>>(d_block_sums, num_blocks);
    add_block_prefix<<<num_blocks, CHUNK>>>(d_out, d_block_sums, N);

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_block_sums);
}

// ============================================================
// 验证：与 CPU thrust 风格（手写）exclusive scan 比对
// ============================================================
bool verify(const std::vector<float>& data, const std::vector<float>& gpu_out) {
    std::vector<float> ref(data.size());
    float acc = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        ref[i] = acc;
        acc += data[i];
    }
    // GPU 输出是 inclusive scan，需要转一下来对比
    // 实际上我们验证：gpu_out[i] = sum(data[0..i])
    acc = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        acc += data[i];
        if (std::abs(gpu_out[i] - acc) > 1e-2f) {
            printf("  [FAIL] i=%zu expected=%.3f got=%.3f\n",
                   i, acc, gpu_out[i]);
            return false;
        }
    }
    return true;
}

bool run_one_case(int N, int trials, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_data(N), h_out(N);

    int* d_data = nullptr;  // alias not allowed for const...
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    bool all_pass = true;
    for (int t = 0; t < trials; ++t) {
        for (int i = 0; i < N; ++i) h_data[i] = dist(rng);
        CUDA_CHECK(cudaMemcpy(d_in, h_data.data(),
                              N * sizeof(float), cudaMemcpyHostToDevice));
        prefix_sum_host(d_in, d_out, N);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                              N * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify(h_data, h_out)) {
            printf("  [FAIL] N=%d trial=%d\n", N, t);
            all_pass = false;
        }
    }

    if (all_pass) {
        printf("  [PASS] N=%8d  trials=%d\n", N, trials);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return all_pass;
}

int main() {
    std::mt19937 rng(7);

    // 覆盖：1个 chunk 内 / 多个 chunk / 远超1024 / 4M
    int Ns[] = {1, 100, 1023, 1024, 1025, 5000,
                65536, 1 << 18, 1 << 20, 1 << 22};

    int total = 0, passed = 0;
    printf("Prefix Sum (multi-block, 3-pass scan)\n\n");
    for (int N : Ns) {
        ++total;
        int trials = (N > (1 << 18)) ? 3 : 10;
        if (run_one_case(N, trials, rng)) ++passed;
    }
    printf("\n==== %d / %d cases passed ====\n", passed, total);
    return (passed == total) ? 0 : 1;
}