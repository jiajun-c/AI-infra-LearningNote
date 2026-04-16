/**
 * 使用 CUDA Driver API 测试 SM 限制
 *
 * 编译：make
 * 运行：./main
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "libsmctrl.h"

// 检查 Driver API 错误
#define CU_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CU error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 检查 Runtime API 错误
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

__global__ void probe_kernel(int* sm_id_array, int num_blocks) {
    if (threadIdx.x == 0 && blockIdx.x < num_blocks) {
        sm_id_array[blockIdx.x] = (int)get_smid();
    }
}

// 统计并打印 SM 使用情况
void print_sm_stats(int* h_sm_ids, int num_blocks, const char* prefix) {
    int sm_seen[256] = {0};
    int sm_count = 0;

    for (int i = 0; i < num_blocks; i++) {
        if (h_sm_ids[i] >= 0 && h_sm_ids[i] < 256 && !sm_seen[h_sm_ids[i]]) {
            sm_seen[h_sm_ids[i]] = 1;
            sm_count++;
        }
    }

    printf("%s实际使用的 SM 数量：%d\n", prefix, sm_count);
    printf("%s涉及的 SM ID: ", prefix);

    int printed = 0;
    for (int sm = 0; sm < 256; sm++) {
        if (sm_seen[sm]) {
            if (printed > 0) printf(", ");
            printf("%d", sm);
            printed++;
            if (printed >= 20) {
                printf(" ... (+%d more)", sm_count - 20);
                break;
            }
        }
    }
    printf("\n");
}

int main() {
    printf("===== CUDA SM 限制测试 =====\n\n");

    // 1. 初始化 Driver API
    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));

    // 获取设备信息
    int total_sms = 0;
    cuDeviceGetAttribute(&total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

    char device_name[256];
    cuDeviceGetName(device_name, sizeof(device_name), device);

    printf("设备：%s\n", device_name);
    printf("SM 总数：%d\n\n", total_sms);

    int num_probe_blocks = 2048;
    int *d_sm_ids, *h_sm_ids;
    CUDA_CHECK(cudaMalloc(&d_sm_ids, num_probe_blocks * sizeof(int)));
    h_sm_ids = (int*)malloc(num_probe_blocks * sizeof(int));

    // 测试 1: 不使用限制（基线）
    printf("=== 测试 1: 无限制（基线）===\n");
    CUDA_CHECK(cudaMemset(d_sm_ids, 0xFF, num_probe_blocks * sizeof(int)));
    probe_kernel<<<num_probe_blocks, 256>>>(d_sm_ids, num_probe_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, num_probe_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    printf("  ");
    print_sm_stats(h_sm_ids, num_probe_blocks, "  ");
    printf("\n");

    // 测试 2: 使用 libsmctrl 限制 SM
    printf("=== 测试 2: 使用 libsmctrl 限制 SM ===\n");

    int test_values[] = {16, 32, 64};
    int num_tests = sizeof(test_values) / sizeof(int);

    for (int i = 0; i < num_tests; i++) {
        int target_sm = test_values[i];

        // 掩码位=1 表示禁用，启用前 N 个 SM
        uint64_t mask = ~((uint64_t)1 << target_sm) + 1;
        libsmctrl_set_next_mask(mask);

        printf("  限制 %2d 个 SM (掩码：0x%016lx): ", target_sm, mask);

        CUDA_CHECK(cudaMemset(d_sm_ids, 0xFF, num_probe_blocks * sizeof(int)));
        probe_kernel<<<num_probe_blocks, 256>>>(d_sm_ids, num_probe_blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, num_probe_blocks * sizeof(int), cudaMemcpyDeviceToHost));

        // 统计实际使用的 SM
        int sm_seen[256] = {0};
        int actual_sm = 0;
        for (int j = 0; j < num_probe_blocks; j++) {
            if (h_sm_ids[j] >= 0 && h_sm_ids[j] < 256 && !sm_seen[h_sm_ids[j]]) {
                sm_seen[h_sm_ids[j]] = 1;
                actual_sm++;
            }
        }

        printf("实际使用 %2d 个 SM\n", actual_sm);

        // 打印前几个 SM ID
        printf("    SM ID: ");
        int printed = 0;
        for (int sm = 0; sm < 256; sm++) {
            if (sm_seen[sm]) {
                if (printed > 0) printf(", ");
                printf("%d", sm);
                printed++;
                if (printed >= 10) break;
            }
        }
        if (actual_sm > 10) printf(" ... (+%d more)", actual_sm - 10);
        printf("\n");
    }

    // 恢复默认
    libsmctrl_set_global_mask(0);

    // 清理
    CUDA_CHECK(cudaFree(d_sm_ids));
    free(h_sm_ids);

    printf("\n===== 测试完成 =====\n");
    return 0;
}
