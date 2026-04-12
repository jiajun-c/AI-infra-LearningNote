#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
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

// ===================== Transpose Kernel V1: 朴素版本 =====================
// 每个线程处理一个元素，存在非合并访问问题
__global__ void transpose_v1(float *in, float *out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int src_idx = row * cols + col;
        int dst_idx = col * rows + row;
        out[dst_idx] = in[src_idx];
    }
}

// ===================== Transpose Kernel V2: 共享内存分块优化 =====================
// 使用共享内存减少全局内存访问，每个块处理一个 TILE x TILE 的子矩阵
#define TILE_SIZE 16

__global__ void transpose_v2(float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 从全局内存加载到共享内存（合并访问）
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }
    __syncthreads();

    // 计算转置后的全局坐标
    int out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int out_col = blockIdx.y * TILE_SIZE + threadIdx.x;

    // 从共享内存写入到全局内存（转置后，线程访问模式变为非合并）
    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

// ===================== Transpose Kernel V3: 共享内存 + 避免 bank conflict =====================
// 通过在共享内存中添加 padding 来避免 bank conflict
#define TILE_SIZE_V3 16
#define PADDED_TILE_SIZE (TILE_SIZE_V3 + 1)  // 添加一个元素的 padding

__global__ void transpose_v3(float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE_SIZE_V3][PADDED_TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE_V3 + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_V3 + threadIdx.x;

    // 从全局内存加载到共享内存（合并访问）
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }
    __syncthreads();

    // 计算转置后的全局坐标
    int out_row = blockIdx.x * TILE_SIZE_V3 + threadIdx.y;
    int out_col = blockIdx.y * TILE_SIZE_V3 + threadIdx.x;

    // 从共享内存写入到全局内存
    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

// ===================== Transpose Kernel V4: 共享内存 + 向量化访存 (float4) =====================
// 使用 float4 向量化加载和存储，每次处理 4 个 float (128-bit)
#define BLOCK_DIM_V4 16
#define VEC_SIZE 4  // 每次处理 4 个 float

__global__ void transpose_v4(float *in, float *out, int rows, int cols) {
    __shared__ float tile[BLOCK_DIM_V4][BLOCK_DIM_V4];

    int col = blockIdx.x * BLOCK_DIM_V4 + threadIdx.x;
    int row = blockIdx.y * BLOCK_DIM_V4 + threadIdx.y;

    // 向量化加载：每个线程加载 4 个连续元素
    if (row < rows && col < cols) {
        // 尝试向量化加载（4 个连续元素）
        if (col + 3 < cols && ((uintptr_t)(&in[row * cols + col]) % 16 == 0)) {
            float4 val = reinterpret_cast<float4*>(&in[row * cols + col])[0];
            tile[threadIdx.y][threadIdx.x] = val.x;
            if (threadIdx.x + 1 < BLOCK_DIM_V4) tile[threadIdx.y][threadIdx.x + 1] = val.y;
            if (threadIdx.x + 2 < BLOCK_DIM_V4) tile[threadIdx.y][threadIdx.x + 2] = val.z;
            if (threadIdx.x + 3 < BLOCK_DIM_V4) tile[threadIdx.y][threadIdx.x + 3] = val.w;
        } else {
            // 标量加载
            tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
        }
    }
    __syncthreads();

    // 转置写入
    int out_row = blockIdx.x * BLOCK_DIM_V4 + threadIdx.x;
    int out_col = blockIdx.y * BLOCK_DIM_V4 + threadIdx.y;

    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.y][threadIdx.x];
    }
}

// ===================== Transpose Kernel V5: 优化的分块版本，32x32 tile =====================
// 使用更大的 32x32 tile，提高 occupancy
#define BLOCK_DIM_V5 32

__global__ void transpose_v5(float *in, float *out, int rows, int cols) {
    __shared__ float tile[BLOCK_DIM_V5][BLOCK_DIM_V5];

    int col = blockIdx.x * BLOCK_DIM_V5 + threadIdx.x;
    int row = blockIdx.y * BLOCK_DIM_V5 + threadIdx.y;

    // 加载
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }
    __syncthreads();

    // 转置写入
    int out_row = blockIdx.x * BLOCK_DIM_V5 + threadIdx.x;
    int out_col = blockIdx.y * BLOCK_DIM_V5 + threadIdx.y;

    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.y][threadIdx.x];
    }
}

// ===================== Transpose Kernel V6: 进一步优化 + 避免 bank conflict =====================
// 使用 padding 避免 bank conflict
#define BLOCK_DIM_V6 32
#define PADDED_SIZE (BLOCK_DIM_V6 + 1)

__global__ void transpose_v6(float *in, float *out, int rows, int cols) {
    __shared__ float tile[BLOCK_DIM_V6][PADDED_SIZE];

    int col = blockIdx.x * BLOCK_DIM_V6 + threadIdx.x;
    int row = blockIdx.y * BLOCK_DIM_V6 + threadIdx.y;

    // 加载
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }
    __syncthreads();

    // 转置写入
    int out_row = blockIdx.x * BLOCK_DIM_V6 + threadIdx.x;
    int out_col = blockIdx.y * BLOCK_DIM_V6 + threadIdx.y;

    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.y][threadIdx.x];
    }
}

// ===================== CPU Reference Implementation =====================
void transpose_cpu(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

// ===================== Correctness Verification =====================
bool verify_correctness(const float *gpu_out, const float *cpu_out,
                        int rows, int cols, float atol = 1e-5f) {
    bool pass = true;
    float max_abs_err = 0.0f;
    int err_count = 0;
    int total = rows * cols;

    for (int i = 0; i < total; i++) {
        float abs_err = fabsf(gpu_out[i] - cpu_out[i]);
        max_abs_err = fmaxf(max_abs_err, abs_err);

        if (abs_err > atol) {
            if (err_count < 10) {
                printf("  Mismatch at [%d][%d]: GPU=%.8f, CPU=%.8f, abs_err=%.2e\n",
                       i / cols, i % cols, gpu_out[i], cpu_out[i], abs_err);
            }
            err_count++;
            pass = false;
        }
    }

    printf("  Max absolute error: %.2e\n", max_abs_err);
    if (err_count > 0) {
        printf("  Total mismatches: %d / %d\n", err_count, total);
    }

    return pass;
}

// ===================== Performance Benchmark =====================
typedef void (*transpose_fn)(float *, float *, int, int);

float benchmark_kernel(transpose_fn kernel, float *d_in, float *d_out,
                       int rows, int cols, dim3 block, dim3 grid,
                       int warmup = 10, int repeat = 100) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat;
}

// ===================== Helper: 获取理论显存带宽 =====================
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
        {32,     32,     "Tiny:     32 x 32"},
        {64,     64,     "Small:    64 x 64"},
        {128,    128,    "Medium:   128 x 128"},
        {256,    256,    "Large:    256 x 256"},
        {512,    512,    "XLarge:   512 x 512"},
        {1024,   1024,   "1K:       1024 x 1024"},
        {2048,   2048,   "2K:       2048 x 2048"},
        {4096,   4096,   "4K:       4096 x 4096"},
        {512,    1024,   "Rect:     512 x 1024"},
        {1024,   512,    "Rect:     1024 x 512"},
        {256,    2048,   "Rect:     256 x 2048"},
        {2048,   256,    "Rect:     2048 x 256"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    // 要测试的 kernel 列表
    struct KernelEntry {
        transpose_fn fn;
        const char *name;
        dim3 block;
        dim3 grid_factor;  // grid 计算因子 (0 表示使用 block 维度)
    };

    KernelEntry kernels[] = {
        {transpose_v1, "V1 (naive)",              {16, 16, 1}, {0, 0, 0}},
        {transpose_v2, "V2 (shared mem)",         {TILE_SIZE, TILE_SIZE, 1}, {TILE_SIZE, TILE_SIZE, 0}},
        {transpose_v3, "V3 (no bank conflict)",   {TILE_SIZE_V3, TILE_SIZE_V3, 1}, {TILE_SIZE_V3, TILE_SIZE_V3, 0}},
        {transpose_v4, "V4 (vec4 + shared)",      {BLOCK_DIM_V4, BLOCK_DIM_V4, 1}, {BLOCK_DIM_V4, BLOCK_DIM_V4, 0}},
        {transpose_v5, "V5 (vec + 8 elems)",      {BLOCK_DIM_V5, BLOCK_DIM_V5, 1}, {BLOCK_DIM_V5, BLOCK_DIM_V5, 0}},
        {transpose_v6, "V6 (vec + padding)",      {BLOCK_DIM_V6, BLOCK_DIM_V6, 1}, {BLOCK_DIM_V6, BLOCK_DIM_V6, 0}},
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
        transpose_cpu(h_in, h_ref, rows, cols);

        // 分配 Device 内存
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice));

        for (int k = 0; k < num_kernels; k++) {
            // 计算 grid 维度
            dim3 block = kernels[k].block;
            dim3 grid;

            if (kernels[k].grid_factor.x == 0) {
                // V1: 基于 block 维度计算 grid
                grid.x = (cols + block.x - 1) / block.x;
                grid.y = (rows + block.y - 1) / block.y;
            } else {
                // V2-V6: 基于 tile_size 计算 grid
                int tile_size = kernels[k].grid_factor.x;
                grid.x = (cols + tile_size - 1) / tile_size;
                grid.y = (rows + tile_size - 1) / tile_size;
            }

            printf("\n  [%s]  block=(%d,%d), grid=(%d,%d)\n",
                   kernels[k].name, block.x, block.y, grid.x, grid.y);

            // ===== 正确性验证 =====
            CUDA_CHECK(cudaMemset(d_out, 0, total * sizeof(float)));
            kernels[k].fn<<<grid, block>>>(d_in, d_out, rows, cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

            printf("  [Correctness]\n");
            bool pass = verify_correctness(h_out, h_ref, rows, cols);
            printf("  Result: %s\n", pass ? "PASS" : "FAIL");

            // ===== 性能测试 =====
            printf("  [Performance]\n");
            float avg_ms = benchmark_kernel(kernels[k].fn, d_in, d_out, rows, cols, block, grid);
            // 带宽计算：读 + 写各一次，共 2 * total * sizeof(float)
            float bandwidth_gb = (2.0f * total * sizeof(float)) / (avg_ms * 1e-3f) / 1e9f;
            printf("  Avg kernel time:   %.4f ms\n", avg_ms);
            printf("  Effective BW:      %.2f GB/s\n", bandwidth_gb);
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
