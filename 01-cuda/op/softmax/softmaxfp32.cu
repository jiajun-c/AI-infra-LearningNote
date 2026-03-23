#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cassert>

using namespace std;

// ===================== CUDA Error Checking =====================
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ===================== Warp-level Primitives =====================
__device__ float warpReduceMax(float val) {
    for (int i = 16; i >= 1; i >>= 1) {
        val = fmaxf(__shfl_xor_sync(0xffffffff, val, i), val);
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int i = 16; i >= 1; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

// ===================== Block-level Reduce =====================
__device__ float blockReduceMax(float val) {
    __shared__ float smem[32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    int numWarps = (blockDim.x + 31) / 32;

    float maxVal = warpReduceMax(val);
    if (laneID == 0) smem[warpID] = maxVal;
    __syncthreads();

    // 只有第一个 warp 参与最终 reduce，未使用的 slot 填 -FLT_MAX
    maxVal = (laneID < numWarps) ? smem[laneID] : -FLT_MAX;
    maxVal = warpReduceMax(maxVal);
    // 广播结果到所有线程
    maxVal = __shfl_sync(0xffffffff, maxVal, 0, 32);
    return maxVal;
}

__device__ float blockReduceSum(float val) {
    __shared__ float smem[32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    int numWarps = (blockDim.x + 31) / 32;

    float sumVal = warpReduceSum(val);
    if (laneID == 0) smem[warpID] = sumVal;
    __syncthreads();

    // 只有第一个 warp 参与最终 reduce，未使用的 slot 填 0
    sumVal = (laneID < numWarps) ? smem[laneID] : 0.0f;
    sumVal = warpReduceSum(sumVal);
    // 广播结果到所有线程
    sumVal = __shfl_sync(0xffffffff, sumVal, 0, 32);
    return sumVal;
}

// ===================== Softmax Kernel (V1: 简化版) =====================
// 每个 block 处理一行，N 为每行的元素个数
// 要求 blockDim.x >= N（适用于 N <= 1024）
__global__ void softmax_v1(float *in, float *out, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int idx = row * N + tid;

    // 超出 N 范围的线程用 -FLT_MAX 填充（不影响 max 和 sum）
    float val = (tid < N) ? in[idx] : -FLT_MAX;

    // Step 1: 求行最大值
    float block_val_max = blockReduceMax(val);

    // Step 2: 计算 exp(val - max)
    float expval = (tid < N) ? expf(val - block_val_max) : 0.0f;

    // Step 3: 求 exp 之和
    float sumVal = blockReduceSum(expval);

    // Step 4: 归一化
    if (tid < N) {
        out[idx] = expval / sumVal;
    }
}

// ===================== Softmax Kernel (V2: 通用版，支持任意 N) =====================
// 每个 block 处理一行，每个线程通过循环（stride loop）处理多个元素
// 适用于任意 N，包括 N > 1024
__global__ void softmax(float *in, float *out, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    float *row_in  = in  + row * N;
    float *row_out = out + row * N;

    // ========== Step 1: 每个线程循环求局部最大值 ==========
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    // Block-level reduce 得到全局最大值
    float row_max = blockReduceMax(local_max);

    // ========== Step 2: 每个线程循环求局部 exp 之和 ==========
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(row_in[i] - row_max);
    }
    // Block-level reduce 得到全局 sum
    float row_sum = blockReduceSum(local_sum);

    // ========== Step 3: 每个线程循环写出归一化结果 ==========
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
    }
}

// ===================== CPU Reference Implementation =====================
void softmax_cpu(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;

        // 找最大值
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; c++) {
            max_val = fmaxf(max_val, row_in[c]);
        }

        // 计算 exp 和 sum
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }

        // 归一化
        for (int c = 0; c < cols; c++) {
            row_out[c] /= sum;
        }
    }
}

// ===================== Correctness Verification =====================
bool verify_correctness(const float *gpu_out, const float *cpu_out,
                        int rows, int cols, float atol = 1e-5f, float rtol = 1e-4f) {
    bool pass = true;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int err_count = 0;

    for (int i = 0; i < rows * cols; i++) {
        float abs_err = fabsf(gpu_out[i] - cpu_out[i]);
        float rel_err = abs_err / (fabsf(cpu_out[i]) + 1e-8f);
        max_abs_err = fmaxf(max_abs_err, abs_err);
        max_rel_err = fmaxf(max_rel_err, rel_err);

        if (abs_err > atol && rel_err > rtol) {
            if (err_count < 10) {
                printf("  Mismatch at [%d][%d]: GPU=%.8f, CPU=%.8f, abs_err=%.2e, rel_err=%.2e\n",
                       i / cols, i % cols, gpu_out[i], cpu_out[i], abs_err, rel_err);
            }
            err_count++;
            pass = false;
        }
    }

    printf("  Max absolute error: %.2e\n", max_abs_err);
    printf("  Max relative error: %.2e\n", max_rel_err);
    if (err_count > 0) {
        printf("  Total mismatches: %d / %d\n", err_count, rows * cols);
    }

    // 额外验证：每行 softmax 之和应为 1.0
    for (int r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_sum += gpu_out[r * cols + c];
        }
        if (fabsf(row_sum - 1.0f) > 1e-3f) {
            printf("  Row %d sum = %.6f (expected 1.0)\n", r, row_sum);
            pass = false;
        }
    }

    return pass;
}

// ===================== Performance Benchmark =====================
typedef void (*softmax_fn)(float *, float *, int);

float benchmark_kernel(softmax_fn kernel, float *d_in, float *d_out,
                       int rows, int cols, int block_size,
                       int warmup = 10, int repeat = 100) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<rows, block_size>>>(d_in, d_out, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        kernel<<<rows, block_size>>>(d_in, d_out, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat;  // 返回平均每次耗时 (ms)
}

// ===================== Helper: 获取理论显存带宽 =====================
// 通过 cudaDeviceGetAttribute 获取 memory clock rate 和 bus width，计算峰值带宽
double get_peak_memory_bandwidth_gbs(int device = 0) {
    int mem_clock_khz = 0;  // kHz
    int bus_width_bits = 0; // bits
    // cudaDevAttrMemoryClockRate 和 cudaDevAttrGlobalMemoryBusWidth
    // 在部分新版 CUDA 中 cudaDeviceProp 移除了 memoryClockRate，但 attribute 查询仍可用
    cudaError_t e1 = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaError_t e2 = cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    if (e1 == cudaSuccess && e2 == cudaSuccess && mem_clock_khz > 0 && bus_width_bits > 0) {
        // DDR: x2, kHz->Hz: x1e3, bits->bytes: /8, 转 GB/s: /1e9
        return 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;
    }
    return 0.0;  // 无法获取时返回 0，跳过利用率计算
}

// ===================== Main =====================
int main(int argc, char **argv) {
    // 打印 GPU 信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw_gbs = get_peak_memory_bandwidth_gbs(0);
    printf("========================================\n");
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d, Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    if (peak_bw_gbs > 0) {
        printf("Memory bandwidth (theoretical): %.1f GB/s\n", peak_bw_gbs);
    }
    printf("========================================\n\n");

    // 测试配置：(rows, cols)
    struct TestConfig {
        int rows;
        int cols;
        const char *desc;
    };

    TestConfig configs[] = {
        {1,      32,    "Tiny:     1 x 32"},
        {1,      128,   "Small:    1 x 128"},
        {64,     128,   "Medium:   64 x 128"},
        {128,    256,   "Large:    128 x 256"},
        {256,    512,   "XLarge:   256 x 512"},
        {512,    1024,  "XXL:      512 x 1024"},
        {1024,   1024,  "1K:       1024 x 1024"},
        // N > 1024 的场景，只有 V2 (stride loop) 能正确处理
        {128,    2048,  ">1K:      128 x 2048"},
        {64,     4096,  ">1K:      64 x 4096"},
        {32,     8192,  ">1K:      32 x 8192"},
        {16,     16384, ">1K:      16 x 16384"},
        {4,      32768, ">1K:      4 x 32768"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    // 要测试的 kernel 列表
    struct KernelEntry {
        softmax_fn fn;
        const char *name;
        bool supports_large;  // 是否支持 N > 1024
    };

    KernelEntry kernels[] = {
        {softmax_v1, "V1 (1 elem/thread)",  false},
        {softmax,    "V2 (stride loop)",     true},
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    for (int t = 0; t < num_configs; t++) {
        int rows = configs[t].rows;
        int cols = configs[t].cols;
        int total = rows * cols;
        printf("========== Test: %s ==========\n", configs[t].desc);

        // 分配 Host 内存
        float *h_in     = (float *)malloc(total * sizeof(float));
        float *h_out    = (float *)malloc(total * sizeof(float));
        float *h_ref    = (float *)malloc(total * sizeof(float));

        // 初始化随机输入（范围 [-5, 5]）
        srand(42);
        for (int i = 0; i < total; i++) {
            h_in[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        }

        // CPU 参考结果
        softmax_cpu(h_in, h_ref, rows, cols);

        // 分配 Device 内存
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice));

        for (int k = 0; k < num_kernels; k++) {
            // V1 不支持 N > 1024，跳过
            if (!kernels[k].supports_large && cols > 1024) {
                printf("\n  [%s] SKIPPED (N=%d > 1024, 不支持)\n", kernels[k].name, cols);
                continue;
            }

            // block_size: V1 取 >= cols 的最小 32 倍数; V2 固定 256 或按需调整
            int block_size;
            if (!kernels[k].supports_large) {
                // V1: blockDim.x >= cols
                block_size = ((cols + 31) / 32) * 32;
                block_size = min(block_size, 1024);
            } else {
                // V2: 用 256 线程即可循环处理任意长度
                block_size = (cols <= 1024) ? min(((cols + 31) / 32) * 32, 1024) : 256;
            }

            printf("\n  [%s]  block_size=%d\n", kernels[k].name, block_size);

            // ===== 正确性验证 =====
            CUDA_CHECK(cudaMemset(d_out, 0, total * sizeof(float)));
            kernels[k].fn<<<rows, block_size>>>(d_in, d_out, cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

            printf("  [Correctness]\n");
            bool pass = verify_correctness(h_out, h_ref, rows, cols);
            printf("  Result: %s\n", pass ? "PASS ✓" : "FAIL ✗");

            // ===== 性能测试 =====
            printf("  [Performance]\n");
            float avg_ms = benchmark_kernel(kernels[k].fn, d_in, d_out, rows, cols, block_size);
            float bandwidth_gb = (2.0f * total * sizeof(float)) / (avg_ms * 1e-3f) / 1e9f;
            float throughput_gflops = (float)(rows) * (3.0f * cols) / (avg_ms * 1e-3f) / 1e9f;
            printf("  Avg kernel time:   %.4f ms\n", avg_ms);
            printf("  Effective BW:      %.2f GB/s\n", bandwidth_gb);
            printf("  Throughput:        %.2f GFLOPS\n", throughput_gflops);
            if (peak_bw_gbs > 0) {
                printf("  Theoretical BW util: %.1f%%\n", bandwidth_gb / peak_bw_gbs * 100.0);
            }
        }
        printf("\n");

        // 释放内存
        free(h_in);
        free(h_out);
        free(h_ref);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("========================================\n");
    printf("All tests completed.\n");
    printf("========================================\n");

    return 0;
}