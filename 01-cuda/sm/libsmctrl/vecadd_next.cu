/**
 * VecAdd 示例 - 使用 libsmctrl_set_next_mask 精确控制 SM 数量
 *
 * 编译命令:
 *   nvcc -o vecadd_next vecadd_next.cu -L/volume/code/jjcheng/libsmctrl -lsmctrl -lcuda -I/volume/code/jjcheng/libsmctrl
 *
 * 运行:
 *   ./vecadd_next           # 使用全部 SM
 *   ./vecadd_next --sm 32   # 限制使用 32 个 SM (启用 SM 0-31)
 *   ./vecadd_next --sm 66   # 限制使用 66 个 SM (启用 SM 0-65)
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

__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// VecAdd kernel: C = A + B
__global__ void vecadd_kernel(const float* A, const float* B, float* C, int n, int* sm_id_array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }

    // 每个 block 的第一个线程记录 SM ID
    if (threadIdx.x == 0) {
        unsigned int smid = get_smid();
        // 使用原子操作记录每个 block 使用的 SM
        if (blockIdx.x < 300) {
            sm_id_array[blockIdx.x] = smid;
        }
    }
}

// 静默版本（不打印）
__global__ void vecadd_kernel_quiet(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int get_sm_count() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("===== GPU 信息 =====\n");
    printf("设备名称：%s\n", prop.name);
    printf("SM 数量：%d\n", prop.multiProcessorCount);
    printf("===================\n\n");
}

void print_sm_stats(int* h_sm_ids, int num_blocks) {
    int sm_counts[256] = {0};
    int min_sm = 255, max_sm = 0;
    int unique_sms = 0;

    for (int i = 0; i < num_blocks; i++) {
        int smid = h_sm_ids[i];
        sm_counts[smid]++;
        if (smid < min_sm) min_sm = smid;
        if (smid > max_sm) max_sm = smid;
    }

    printf("===== SM 使用统计 =====\n");
    printf("Blocks 总数：%d\n", num_blocks);

    // 统计使用的 SM 数量
    for (int i = 0; i < 256; i++) {
        if (sm_counts[i] > 0) unique_sms++;
    }
    printf("使用的 SM 数量：%d\n", unique_sms);
    printf("SM ID 范围：%d - %d\n", min_sm, max_sm);

    // 打印每个 SM 的 block 数
    printf("\nSM 分布:\n");
    for (int i = 0; i < 256; i++) {
        if (sm_counts[i] > 0) {
            printf("  SM %3d: %d blocks\n", i, sm_counts[i]);
        }
    }
}

int main(int argc, char** argv) {
    int sm_limit = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc) {
            sm_limit = atoi(argv[++i]);
            break;
        }
    }

    print_device_info();

    int actual_sm = get_sm_count();
    printf("目标 SM 数量：%d\n\n", sm_limit > 0 ? sm_limit : actual_sm);

    // Kernel 配置：约 300 个 blocks
    int threads_per_block = 256;
    int num_blocks = 300;
    int array_size = num_blocks * threads_per_block;

    printf("Kernel 配置:\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads/block: %d\n", threads_per_block);
    printf("  Array size: %d\n\n", array_size);

    // 分配设备内存
    size_t bytes = array_size * sizeof(float);
    float *d_A, *d_B, *d_C;
    int *d_sm_ids;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_sm_ids, num_blocks * sizeof(int)));

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

    // 初始化 SM ID 数组
    CUDA_CHECK(cudaMemset(d_sm_ids, 0, num_blocks * sizeof(int)));

    // 应用 SM 限制
    if (sm_limit > 0 && sm_limit <= 64) {
        uint64_t mask = ~((uint64_t)1 << sm_limit) + 1;
        libsmctrl_set_next_mask(mask);
        printf("设置 next SM mask: %d (掩码：0x%016lx)\n", sm_limit, mask);
    } else if (sm_limit > 64) {
        fprintf(stderr, "错误：--sm 最多支持 64 (受限于 64 位掩码)\n");
        return 1;
    } else {
        printf("不使用 SM 限制 (使用全部 %d 个 SM)\n", actual_sm);
    }
    printf("\n");

    // 运行 kernel
    printf("===== 运行 Kernel =====\n");
    vecadd_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, array_size, d_sm_ids);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝 SM ID 结果
    int *h_sm_ids = (int*)malloc(num_blocks * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    // 打印前 20 个 block 的 SM ID
    printf("\n前 20 个 block 的 SM ID:\n");
    for (int i = 0; i < 20 && i < num_blocks; i++) {
        printf("  Block %3d -> SM %d\n", i, h_sm_ids[i]);
    }
    printf("\n");

    // 打印统计信息
    print_sm_stats(h_sm_ids, num_blocks);
    printf("\n");

    // 性能测试
    printf("===== 性能测试 =====\n");
    int warmup_iters = 10;
    int test_iters = 20;

    for (int i = 0; i < warmup_iters; i++) {
        vecadd_kernel_quiet<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, array_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

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

    double bytes_rw = (double)array_size * 4.0 * 3.0;  // 读 A, B + 写 C
    double bandwidth = (bytes_rw / (avg_ms / 1000.0)) / 1e9;
    printf("带宽：%.2f GB/s\n", bandwidth);

    // 验证
    float *h_C = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < 10; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            errors++;
        }
    }
    printf("结果验证：%s\n", errors == 0 ? "PASS" : "FAIL");

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_sm_ids));
    free(h_sm_ids);
    free(h_C);

    return 0;
}
