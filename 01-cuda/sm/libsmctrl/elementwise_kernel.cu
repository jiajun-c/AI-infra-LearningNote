/**
 * 完整的 CUDA kernel 示例，展示如何使用 libsmctrl 限制 SM 使用量
 *
 * 编译命令：
 *   nvcc -o elementwise_kernel elementwise_kernel.cu -L/volume/code/jjcheng/libsmctrl -lsmctrl -lcuda -I/volume/code/jjcheng/libsmctrl
 *
 * 运行：
 *   ./elementwise_kernel              # 不限制 SM（使用全部 132 个）
 *   ./elementwise_kernel --sm 64      # 限制使用 64 个 SM
 *   ./elementwise_kernel --global 64  # 全局限制 64 个 SM
 *   ./elementwise_kernel --bench      # 运行完整基准测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "libsmctrl.h"

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 简单的 element-wise kernel：y = a * x + b
__global__ void elementwise_kernel(float *x, float *y, float a, float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + b;
    }
}

// 计算密集型 kernel：模拟更复杂的计算
__global__ void compute_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // 模拟多次计算
        for (int i = 0; i < 10; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        y[idx] = val;
    }
}

// 获取当前设备 SM 数量
int get_sm_count() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

// 打印设备信息
void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("===== GPU 信息 =====\n");
    printf("设备名称：%s\n", prop.name);
    printf("SM 数量：%d\n", prop.multiProcessorCount);
    printf("计算能力：%d.%d\n", prop.major, prop.minor);
    printf("===================\n\n");
}

// 设置 SM 限制（支持超过 64 个 SM 的 GPU）
void set_sm_limit_next(int sm_count) {
    int actual_sm_count = get_sm_count();

    if (sm_count <= 0 || sm_count >= actual_sm_count) {
        libsmctrl_set_next_mask(0);
        printf("SM 限制：无 (使用全部 %d 个 SM)\n", actual_sm_count);
        return;
    }

    // 掩码位=1 表示禁用该 SM
    // 要启用前 N 个 SM，需要禁用 N 以上的所有 SM
    if (sm_count <= 64) {
        uint64_t mask = ~((uint64_t)1 << sm_count) + 1;
        libsmctrl_set_next_mask(mask);
        printf("设置 SM 限制：%d (64 位掩码：0x%016lx)\n", sm_count, mask);
    } else {
        // 超过 64 个 SM，使用 stream mask ext
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        uint128_t mask = ~(~(uint128_t)0 << sm_count);
        libsmctrl_set_stream_mask_ext((void*)stream, mask);
        printf("设置 SM 限制：%d (128 位掩码，通过 stream)\n", sm_count);
        // 注意：stream mask 会持续影响该 stream 上的所有 kernel
    }
}

// 设置全局 SM 限制
void set_sm_limit_global(int sm_count) {
    int actual_sm_count = get_sm_count();

    if (sm_count <= 0 || sm_count >= actual_sm_count) {
        libsmctrl_set_global_mask(0);
        printf("设置全局 SM 限制：无 (使用全部 %d 个 SM)\n", actual_sm_count);
        return;
    }

    if (sm_count <= 64) {
        uint64_t mask = ~((uint64_t)1 << sm_count) + 1;
        libsmctrl_set_global_mask(mask);
        printf("设置全局 SM 限制：%d (64 位掩码：0x%016lx)\n", sm_count, mask);
    } else {
        // 超过 64 个 SM，使用全局 mask（只影响低 64 个 SM）
        // 注意：libsmctrl 没有 global_mask_ext，对于>64 SM 的 GPU，
        // 建议使用 stream_mask_ext 来限制
        fprintf(stderr, "警告：全局 mask 最多支持 64 个 SM。对于 %d 个 SM，请使用 stream 模式。\n", sm_count);
        uint64_t mask = ~((uint64_t)1 << 64) + 1;  // 禁用 64 以上的所有 SM
        libsmctrl_set_global_mask(mask);
        printf("设置全局 SM 限制：64 (禁用 64 以上的 SM)\n");
    }
}

// 基准测试
void benchmark(const char* name, int sm_limit, int array_size) {
    // 分配内存
    float *x, *y;
    size_t bytes = array_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));

    // 初始化数据
    float *host_x = (float*)malloc(bytes);
    for (int i = 0; i < array_size; i++) {
        host_x[i] = i * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(x, host_x, bytes, cudaMemcpyHostToDevice));
    free(host_x);

    // 配置 kernel
    int threads_per_block = 256;
    int num_blocks = (array_size + threads_per_block - 1) / threads_per_block;

    // 预热
    for (int i = 0; i < 5; i++) {
        elementwise_kernel<<<num_blocks, threads_per_block>>>(x, y, 2.0f, 1.0f, array_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 设置 SM 限制
    set_sm_limit_next(sm_limit);

    // 正式测试
    int iterations = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        elementwise_kernel<<<num_blocks, threads_per_block>>>(x, y, 2.0f, 1.0f, array_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    printf("%-20s: %.3f ms (%.3f ms/iter)\n", name, elapsed_ms, elapsed_ms / iterations);

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
}

void print_usage(const char* prog) {
    printf("用法：%s [选项]\n", prog);
    printf("选项:\n");
    printf("  --sm <N>       下一个 kernel 使用 N 个 SM\n");
    printf("  --global <N>   全局默认使用 N 个 SM\n");
    printf("  --no-limit     不使用 SM 限制（默认）\n");
    printf("  --bench        运行完整基准测试（测试多个 SM 数量）\n");
    printf("  --help         显示此帮助信息\n");
}

int main(int argc, char** argv) {
    int sm_limit = -1;
    int global_limit = -1;
    int run_bench = 0;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc) {
            sm_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--global") == 0 && i + 1 < argc) {
            global_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-limit") == 0) {
            sm_limit = -1;
            global_limit = -1;
        } else if (strcmp(argv[i], "--bench") == 0) {
            run_bench = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // 打印设备信息
    print_device_info();

    int array_size = 100000000;  // 1 亿元素

    if (run_bench) {
        // 基准测试模式：测试多个 SM 配置
        printf("===== 基准测试 =====\n");
        printf("数组大小：%d 元素 (%.2f MB)\n\n", array_size, array_size * sizeof(float) / 1e6);

        int actual_sm = get_sm_count();
        int sm_counts[] = {132, 96, 64, 48, 32, 24, 16, 8};
        int num_configs = sizeof(sm_counts) / sizeof(sm_counts[0]);

        for (int i = 0; i < num_configs; i++) {
            if (sm_counts[i] > actual_sm) continue;  // 跳过超过实际 SM 数量的配置

            printf("--- 配置 %d ---\n", i + 1);
            benchmark("Element-wise", sm_counts[i], array_size);
            printf("\n");
        }
    } else {
        // 单次运行模式
        printf("===== 运行测试 =====\n");
        printf("数组大小：%d 元素 (%.2f MB)\n\n", array_size, array_size * sizeof(float) / 1e6);

        // 应用全局限制
        if (global_limit > 0) {
            set_sm_limit_global(global_limit);
        }

        // 运行测试
        benchmark("Element-wise", sm_limit, array_size);
    }

    // 恢复默认（使用全部 SM）
    libsmctrl_set_global_mask(0);

    printf("\n完成。\n");
    return 0;
}
