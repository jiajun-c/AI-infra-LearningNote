#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <algorithm>
#include <vector>

using namespace std;

// ===================== CUDA 错误检查宏 =====================
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ===================== Warp-level Reduce =====================
__device__ float warpReduceMax(float val) {
    for (int i = 16; i >= 1; i >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

// ===================== Block-level Reduce (修复版) =====================
// 原始版本存在 bug：第一个 warp 做最终 reduce 时没有从 smem 读取，
// 且未使用的 lane 没有填充 -FLT_MAX，导致结果错误。
__device__ float blockReduceMax(float val) {
    __shared__ float smem[32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    int numWarps = (blockDim.x + 31) / 32;

    // 每个 warp 先内部 reduce
    float maxVal = warpReduceMax(val);
    // 每个 warp 的 lane 0 写入 shared memory
    if (laneID == 0) smem[warpID] = maxVal;
    __syncthreads();

    // 只有第一个 warp 参与最终 reduce
    // 未使用的 slot 填充 -FLT_MAX，避免影响结果
    maxVal = (laneID < numWarps) ? smem[laneID] : -FLT_MAX;
    maxVal = warpReduceMax(maxVal);
    return maxVal;
}

// ===================== ReduceMax Kernel (修复版) =====================
// 每个 block 负责对整个输入数组做 reduce，最终由 block 0 的 thread 0 输出结果
// 当使用多个 block 时，需要二阶段 reduce
__global__ void reduceMax(float* in, float* out, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    // 每个线程通过 stride loop 遍历自己负责的元素
    float maxVal = -FLT_MAX;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        maxVal = fmaxf(maxVal, in[i]);
    }

    // Block 内 reduce
    maxVal = blockReduceMax(maxVal);

    // 每个 block 输出一个部分结果
    if (tid == 0) {
        out[blockIdx.x] = maxVal;
    }
}

// ===================== 原始版本（含 bug，用于对比） =====================
__device__ float warpReduceMax_orig(float val) {
    for (int i = 16; i >= 1; i >>= 1) {
        val = fmax(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

__device__ float blockReduceMax_orig(float val) {
    __shared__ float smem[32];
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    float maxVal = warpReduceMax_orig(val);
    if (laneID == 0) smem[warpID] = maxVal;
    __syncthreads();
    // BUG: 没有从 smem 读取，而是继续用自己的 maxVal
    // BUG: 未使用的 lane 没有填充 -FLT_MAX
    maxVal = warpReduceMax_orig(maxVal);
    return maxVal;
}

__global__ void reduceMax_orig(float* in, float* out, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    float maxVal = idx < N ? in[idx] : -1e9;
    for (int i = tid; i < N; i += blockDim.x) {
        maxVal = max(maxVal, in[blockDim.x * blockIdx.x + i]);
    }
    maxVal = blockReduceMax_orig(maxVal);
    if (tid == 0) {
        out[blockIdx.x] = maxVal;
    }
}

// ===================== CPU 参考实现 =====================
float reduceMax_cpu(const float* data, int N) {
    float maxVal = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        maxVal = fmaxf(maxVal, data[i]);
    }
    return maxVal;
}

// ===================== 正确性验证 =====================
struct TestResult {
    bool pass;
    float gpu_val;
    float cpu_val;
    float abs_err;
};

TestResult verify_single(float gpu_val, float cpu_val, float atol = 1e-5f) {
    TestResult res;
    res.gpu_val = gpu_val;
    res.cpu_val = cpu_val;
    res.abs_err = fabsf(gpu_val - cpu_val);
    res.pass = (res.abs_err <= atol);
    return res;
}

// ===================== 性能测试 =====================
typedef void (*reduce_fn)(float*, float*, int);

float benchmark_kernel(reduce_fn kernel, float* d_in, float* d_out,
                       int N, int grid_size, int block_size,
                       int warmup = 20, int repeat = 200) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid_size, block_size>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        kernel<<<grid_size, block_size>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat;
}

// ===================== 获取理论显存带宽 =====================
double get_peak_memory_bandwidth_gbs(int device = 0) {
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaError_t e2 = cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    if (e1 == cudaSuccess && e2 == cudaSuccess && mem_clock_khz > 0 && bus_width_bits > 0) {
        return 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;
    }
    return 0.0;
}

// ===================== Main =====================
int main(int argc, char** argv) {
    // 打印 GPU 信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw_gbs = get_peak_memory_bandwidth_gbs(0);

    printf("========================================\n");
    printf("  ReduceMax 正确性 & 性能测试\n");
    printf("========================================\n");
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d, Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    if (peak_bw_gbs > 0) {
        printf("Memory bandwidth (theoretical): %.1f GB/s\n", peak_bw_gbs);
    }
    printf("========================================\n\n");

    // ==================== 第一部分：正确性测试 ====================
    printf("╔══════════════════════════════════════╗\n");
    printf("║       第一部分：正确性测试           ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    struct CorrectnessConfig {
        int N;
        const char* desc;
    };

    CorrectnessConfig correctness_tests[] = {
        {1,         "N=1       (单元素)"},
        {32,        "N=32      (单 warp)"},
        {64,        "N=64      (2 warps)"},
        {128,       "N=128     (4 warps)"},
        {256,       "N=256     (8 warps)"},
        {512,       "N=512     (16 warps)"},
        {1024,      "N=1024    (32 warps, 满 block)"},
        {1023,      "N=1023    (非对齐)"},
        {2048,      "N=2048    (2K)"},
        {4096,      "N=4096    (4K)"},
        {8192,      "N=8192    (8K)"},
        {65536,     "N=65536   (64K)"},
        {1048576,   "N=1048576 (1M)"},
        {16777216,  "N=16777216(16M)"},
    };
    int num_correctness = sizeof(correctness_tests) / sizeof(correctness_tests[0]);

    int total_pass = 0;
    int total_tests = 0;

    for (int t = 0; t < num_correctness; t++) {
        int N = correctness_tests[t].N;
        printf("--- %s ---\n", correctness_tests[t].desc);

        // 分配内存
        float* h_in = (float*)malloc(N * sizeof(float));

        // 初始化随机输入 [-100, 100]
        srand(42 + t);  // 每个测试不同的种子
        for (int i = 0; i < N; i++) {
            h_in[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        }

        // 随机位置放一个已知最大值，便于验证
        float known_max = 999.99f;
        h_in[rand() % N] = known_max;

        // CPU 参考结果
        float cpu_max = reduceMax_cpu(h_in, N);

        // Device 内存
        float* d_in;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        // ---------- 测试修复版 ----------
        int block_size = 256;
        int grid_size = min((N + block_size - 1) / block_size, 1024);

        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, grid_size * sizeof(float)));

        // 第一阶段
        reduceMax<<<grid_size, block_size>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 如果使用多个 block，需要二阶段 reduce
        if (grid_size > 1) {
            float* d_out2;
            CUDA_CHECK(cudaMalloc(&d_out2, sizeof(float)));
            reduceMax<<<1, block_size>>>(d_out, d_out2, grid_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            float gpu_max;
            CUDA_CHECK(cudaMemcpy(&gpu_max, d_out2, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_out2));

            TestResult res = verify_single(gpu_max, cpu_max);
            printf("  [修复版] GPU=%.4f, CPU=%.4f, abs_err=%.2e => %s\n",
                   res.gpu_val, res.cpu_val, res.abs_err,
                   res.pass ? "PASS ✓" : "FAIL ✗");
            if (res.pass) total_pass++;
            total_tests++;
        } else {
            float gpu_max;
            CUDA_CHECK(cudaMemcpy(&gpu_max, d_out, sizeof(float), cudaMemcpyDeviceToHost));

            TestResult res = verify_single(gpu_max, cpu_max);
            printf("  [修复版] GPU=%.4f, CPU=%.4f, abs_err=%.2e => %s\n",
                   res.gpu_val, res.cpu_val, res.abs_err,
                   res.pass ? "PASS ✓" : "FAIL ✗");
            if (res.pass) total_pass++;
            total_tests++;
        }

        // ---------- 测试原始版（仅单 block 场景） ----------
        if (N <= 1024) {
            float* d_out_orig;
            CUDA_CHECK(cudaMalloc(&d_out_orig, sizeof(float)));
            reduceMax_orig<<<1, N <= 32 ? 32 : ((N + 31) / 32) * 32>>>(d_in, d_out_orig, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            float gpu_max_orig;
            CUDA_CHECK(cudaMemcpy(&gpu_max_orig, d_out_orig, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_out_orig));

            TestResult res_orig = verify_single(gpu_max_orig, cpu_max);
            printf("  [原始版] GPU=%.4f, CPU=%.4f, abs_err=%.2e => %s\n",
                   res_orig.gpu_val, res_orig.cpu_val, res_orig.abs_err,
                   res_orig.pass ? "PASS ✓" : "FAIL ✗");
            if (!res_orig.pass) {
                printf("           ^ 原始版 blockReduceMax 存在 bug，结果可能不正确\n");
            }
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h_in);
        printf("\n");
    }

    printf("─────────────────────────────────────────\n");
    printf("正确性总结: %d / %d PASSED\n", total_pass, total_tests);
    printf("─────────────────────────────────────────\n\n");

    // ==================== 第二部分：边界条件测试 ====================
    printf("╔══════════════════════════════════════╗\n");
    printf("║       第二部分：边界条件测试         ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    // 测试 1: 全部相同值
    {
        int N = 4096;
        float* h_in = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = 42.0f;

        float* d_in;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        float* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

        reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
        reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  全部相同值 (42.0):  GPU=%.4f => %s\n",
               result, fabsf(result - 42.0f) < 1e-5f ? "PASS ✓" : "FAIL ✗");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_partial));
        free(h_in);
    }

    // 测试 2: 全部负值
    {
        int N = 4096;
        float* h_in = (float*)malloc(N * sizeof(float));
        srand(123);
        for (int i = 0; i < N; i++) h_in[i] = -((float)rand() / RAND_MAX) * 100.0f - 1.0f;

        float cpu_max = reduceMax_cpu(h_in, N);

        float* d_in;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        float* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

        reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
        reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  全部负值:           GPU=%.4f, CPU=%.4f => %s\n",
               result, cpu_max, fabsf(result - cpu_max) < 1e-5f ? "PASS ✓" : "FAIL ✗");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_partial));
        free(h_in);
    }

    // 测试 3: 最大值在最后一个位置
    {
        int N = 4096;
        float* h_in = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        h_in[N - 1] = 999.0f;

        float* d_in;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        float* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

        reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
        reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  最大值在末尾:       GPU=%.4f => %s\n",
               result, fabsf(result - 999.0f) < 1e-5f ? "PASS ✓" : "FAIL ✗");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_partial));
        free(h_in);
    }

    // 测试 4: 递增序列（最大值在最后）
    {
        int N = 8192;
        float* h_in = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = (float)i;

        float* d_in;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        float* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

        reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
        reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        float expected = (float)(N - 1);
        printf("  递增序列 [0..%d]:   GPU=%.4f => %s\n",
               N - 1, result, fabsf(result - expected) < 1e-1f ? "PASS ✓" : "FAIL ✗");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_partial));
        free(h_in);
    }

    // 测试 5: 包含特殊浮点值
    {
        int N = 1024;
        float* h_in = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = -50.0f;
        h_in[100] = FLT_MAX;
        h_in[200] = -FLT_MAX;

        float* d_in;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        reduceMax<<<1, 256>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  含 FLT_MAX/-FLT_MAX: GPU=%.4e => %s\n",
               result, result == FLT_MAX ? "PASS ✓" : "FAIL ✗");

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h_in);
    }

    printf("\n");

    // ==================== 第三部分：性能测试 ====================
    printf("╔══════════════════════════════════════╗\n");
    printf("║       第三部分：性能测试             ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    struct PerfConfig {
        int N;
        const char* desc;
    };

    PerfConfig perf_tests[] = {
        {1024,      "1K    "},
        {4096,      "4K    "},
        {16384,     "16K   "},
        {65536,     "64K   "},
        {262144,    "256K  "},
        {1048576,   "1M    "},
        {4194304,   "4M    "},
        {16777216,  "16M   "},
        {67108864,  "64M   "},
    };
    int num_perf = sizeof(perf_tests) / sizeof(perf_tests[0]);

    // 表头
    printf("  %-8s | %-12s | %-12s | %-14s | %-12s\n",
           "N", "Kernel(us)", "Total(us)", "Eff BW(GB/s)", "BW Util(%)");
    printf("  ─────────┼──────────────┼──────────────┼────────────────┼─────────────\n");

    for (int t = 0; t < num_perf; t++) {
        int N = perf_tests[t].N;
        int block_size = 256;
        int grid_size = min((N + block_size - 1) / block_size, 1024);

        // 分配输入
        float* h_in = (float*)malloc(N * sizeof(float));
        srand(42);
        for (int i = 0; i < N; i++) {
            h_in[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        }

        float* d_in;
        float* d_partial;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        // Warmup
        for (int i = 0; i < 20; i++) {
            reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
            if (grid_size > 1) {
                reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark: 第一阶段 kernel
        float stage1_ms = benchmark_kernel(reduceMax, d_in, d_partial,
                                           N, grid_size, block_size, 20, 200);

        // Benchmark: 完整二阶段
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int repeat = 200;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
            if (grid_size > 1) {
                reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_total_ms = total_ms / repeat;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // 计算有效带宽：reduce 只读取 N 个 float
        float read_bytes = (float)N * sizeof(float);
        float eff_bw_gbs = read_bytes / (avg_total_ms * 1e-3f) / 1e9f;
        float bw_util = peak_bw_gbs > 0 ? eff_bw_gbs / peak_bw_gbs * 100.0f : 0.0f;

        printf("  %-8s | %10.2f   | %10.2f   | %12.2f   | %10.1f\n",
               perf_tests[t].desc,
               stage1_ms * 1000.0f,
               avg_total_ms * 1000.0f,
               eff_bw_gbs,
               bw_util);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_out));
        free(h_in);
    }

    printf("\n");

    // ==================== 第四部分：不同 Block Size 对比 ====================
    printf("╔══════════════════════════════════════╗\n");
    printf("║  第四部分：Block Size 对性能的影响   ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    {
        int N = 16777216;  // 16M
        int block_sizes[] = {32, 64, 128, 256, 512, 1024};
        int num_bs = sizeof(block_sizes) / sizeof(block_sizes[0]);

        float* h_in = (float*)malloc(N * sizeof(float));
        srand(42);
        for (int i = 0; i < N; i++) {
            h_in[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        }

        float* d_in;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        printf("  N = %d (16M)\n\n", N);
        printf("  %-12s | %-10s | %-12s | %-14s | %-12s\n",
               "Block Size", "Grid Size", "Total(us)", "Eff BW(GB/s)", "BW Util(%)");
        printf("  ─────────────┼────────────┼──────────────┼────────────────┼─────────────\n");

        for (int b = 0; b < num_bs; b++) {
            int block_size = block_sizes[b];
            int grid_size = min((N + block_size - 1) / block_size, 1024);

            float* d_partial;
            float* d_out;
            CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

            // Warmup
            for (int i = 0; i < 20; i++) {
                reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
                if (grid_size > 1) {
                    reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            int repeat = 200;
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < repeat; i++) {
                reduceMax<<<grid_size, block_size>>>(d_in, d_partial, N);
                if (grid_size > 1) {
                    reduceMax<<<1, block_size>>>(d_partial, d_out, grid_size);
                }
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_ms;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            float avg_ms = total_ms / repeat;

            float eff_bw_gbs = (float)N * sizeof(float) / (avg_ms * 1e-3f) / 1e9f;
            float bw_util = peak_bw_gbs > 0 ? eff_bw_gbs / peak_bw_gbs * 100.0f : 0.0f;

            printf("  %-12d | %-10d | %10.2f   | %12.2f   | %10.1f\n",
                   block_size, grid_size,
                   avg_ms * 1000.0f,
                   eff_bw_gbs,
                   bw_util);

            CUDA_CHECK(cudaFree(d_partial));
            CUDA_CHECK(cudaFree(d_out));
        }

        CUDA_CHECK(cudaFree(d_in));
        free(h_in);
    }

    printf("\n========================================\n");
    printf("  所有测试完成!\n");
    printf("========================================\n");

    return 0;
}
