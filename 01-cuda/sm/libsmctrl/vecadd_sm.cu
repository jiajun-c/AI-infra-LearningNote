/**
 * 简单的 VecAdd 示例，使用 libsmctrl 控制 SM 数量
 *
 * 编译命令:
 *   nvcc -o vecadd_sm vecadd_sm.cu -L/volume/code/jjcheng/libsmctrl -lsmctrl -lcuda -I/volume/code/jjcheng/libsmctrl
 *
 * 运行:
 *   ./vecadd_sm           # 使用全部 SM
 *   ./vecadd_sm --sm 32   # 限制使用 32 个 SM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "libsmctrl.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 获取 SMID 的内联汇编
__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// VecAdd kernel: C = A + B
__global__ void vecadd_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }

    // 每个 block 的第一个线程打印 SM ID
    if (threadIdx.x == 0) {
        unsigned int smid = get_smid();
        printf("[blockIdx.x=%d, SM=%u]\n", blockIdx.x, smid);
    }
}

// 简化的 VecAdd kernel（减少打印输出）
__global__ void vecadd_kernel_quiet(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// 获取 SM 数量
int get_sm_count() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

// 设置全局 SM 限制
void set_global_sm_limit(int sm_count) {
    int actual_sm = get_sm_count();

    if (sm_count <= 0 || sm_count >= actual_sm) {
        libsmctrl_set_global_mask(0);
        printf("SM 限制：无 (使用全部 %d 个 SM)\n", actual_sm);
        return;
    }

    // 掩码位=1 表示禁用该 SM
    // 启用前 N 个 SM，需要禁用 N 以上的所有 SM
    uint64_t mask = ~((uint64_t)1 << sm_count) + 1;
    libsmctrl_set_global_mask(mask);
    printf("设置全局 SM 限制：%d (掩码：0x%016lx)\n", sm_count, mask);
}

void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("===== GPU 信息 =====\n");
    printf("设备名称：%s\n", prop.name);
    printf("SM 数量：%d\n", prop.multiProcessorCount);
    printf("计算能力：%d.%d\n", prop.major, prop.minor);
    printf("===================\n\n");
}

int main(int argc, char** argv) {
    int sm_limit = -1;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc) {
            sm_limit = atoi(argv[++i]);
            break;
        }
    }

    // 打印设备信息
    print_device_info();

    int actual_sm = get_sm_count();
    printf("目标 SM 数量：%d (默认：全部 %d 个)\n\n", sm_limit > 0 ? sm_limit : actual_sm, actual_sm);

    // 应用全局 SM 限制
    set_global_sm_limit(sm_limit);

    // 问题配置：约 300 个 blocks
    int threads_per_block = 256;
    int num_blocks = 300;  // 大约 300 个 blocks
    int array_size = num_blocks * threads_per_block;  // 76800 个元素

    printf("Kernel 配置:\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads/block: %d\n", threads_per_block);
    printf("  Array size: %d\n", array_size);
    printf("\n");

    // 分配内存
    size_t bytes = array_size * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 初始化数据
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    for (int i = 0; i < array_size; i++) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 0.3f;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    free(h_A);
    free(h_B);

    // 第一次运行：打印 SM ID
    printf("===== 第一次运行 (打印 SM ID) =====\n");
    vecadd_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, array_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // 第二次运行：性能测试（不打印）
    printf("===== 性能测试 (不打印) =====\n");
    int warmup_iters = 10;
    int test_iters = 20;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        vecadd_kernel_quiet<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, array_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式测试
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < test_iters; i++) {
        vecadd_kernel_quiet<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, array_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / test_iters;

    printf("平均时间：%.4f ms\n", avg_ms);

    // 计算带宽
    double bytes_read = (double)array_size * 4.0 * 2.0;  // 读 A 和 B
    double bytes_write = (double)array_size * 4.0;        // 写 C
    double total_bytes = bytes_read + bytes_write;
    double bandwidth = (total_bytes / (avg_ms / 1000.0)) / 1e9;
    printf("带宽：%.2f GB/s\n", bandwidth);

    // 验证结果
    float *h_C = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < 10; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            printf("验证失败 @ index %d: 期望=%.4f, 实际=%.4f\n", i, expected, h_C[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("结果验证：PASS\n");
    } else {
        printf("结果验证：FAIL (%d 个错误)\n", errors);
    }

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_C);

    // 恢复默认
    libsmctrl_set_global_mask(0);

    return 0;
}
