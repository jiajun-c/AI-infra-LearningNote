#include <stdio.h>
#include <stdlib.h>
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

__global__ void probe_kernel(int* sm_id_array, int num_blocks) {
    if (threadIdx.x == 0 && blockIdx.x < num_blocks) {
        sm_id_array[blockIdx.x] = (int)get_smid();
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int total_sms = prop.multiProcessorCount;

    printf("===== H100 SM 掩码校准 =====\n");
    printf("GPU: %s\n", prop.name);
    printf("SM 总数：%d (预计对应 %d 个 TPC)\n\n", total_sms, total_sms / 2);

    // 设置洪水探测数量 (保证能覆盖所有激活的 SM)
    int num_probe_blocks = 2048;
    int *d_sm_ids, *h_sm_ids;
    CUDA_CHECK(cudaMalloc(&d_sm_ids, num_probe_blocks * sizeof(int)));
    h_sm_ids = (int*)malloc(num_probe_blocks * sizeof(int));

    printf("测试单个掩码位 (每次只启用1个掩码位):\n");
    printf("------------------------------------------------------\n");

    for (int bit = 0; bit < 64; bit++) {
        // [修复3]: 正确的单一位启用
        uint64_t mask = (1ULL << bit);

        
        // [修复1]: 用 0xFF 填充，即初始值为 -1，避免 0 被误判为 SM 0
        CUDA_CHECK(cudaMemset(d_sm_ids, 0xFF, num_probe_blocks * sizeof(int)));

        // [修复2]: 启动足够的 block 填满 GPU
        libsmctrl_set_next_mask(mask);
        probe_kernel<<<num_probe_blocks, 256>>>(d_sm_ids, num_probe_blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, num_probe_blocks * sizeof(int), cudaMemcpyDeviceToHost));

        int sm_seen[256] = {0};
        for (int i = 0; i < num_probe_blocks; i++) {
            if (h_sm_ids[i] >= 0 && h_sm_ids[i] < 256) {
                sm_seen[h_sm_ids[i]] = 1;
            }
        }

        printf("掩码位 %2d (0x%016lx) -> 启用的 SM: ", bit, mask);
        int count = 0;
        for (int sm = 0; sm < 256; sm++) {
            if (sm_seen[sm]) {
                if (count > 0) printf(", ");
                printf("%d", sm);
                count++;
            }
        }
        if (count == 0) printf("(无)");
        printf("\n");
    }

    printf("\n\n测试连续掩码 (启用前 N 个掩码位):\n");
    printf("------------------------------------------------------\n");

    int test_values[] = {8, 16, 24, 32, 48, 64};
    int num_tests = sizeof(test_values) / sizeof(int);

    for (int i = 0; i < num_tests; i++) {
        int n = test_values[i];
        
        // [修复3]: 正确的前 N 位掩码生成
        uint64_t mask = (n >= 64) ? ~0ULL : ((1ULL << n) - 1);

        CUDA_CHECK(cudaMemset(d_sm_ids, 0xFF, num_probe_blocks * sizeof(int)));
        libsmctrl_set_next_mask(mask);
        probe_kernel<<<num_probe_blocks, 256>>>(d_sm_ids, num_probe_blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, num_probe_blocks * sizeof(int), cudaMemcpyDeviceToHost));

        int sm_seen[256] = {0};
        int sm_count = 0;
        for (int j = 0; j < num_probe_blocks; j++) {
            if (h_sm_ids[j] >= 0 && h_sm_ids[j] < 256 && !sm_seen[h_sm_ids[j]]) {
                sm_seen[h_sm_ids[j]] = 1;
                sm_count++;
            }
        }

        printf("启用前 %2d 位 -> 使用的 SM 数量：%2d, SM ID: ", n, sm_count);

        int printed = 0;
        for (int sm = 0; sm < 256; sm++) {
            if (sm_seen[sm]) {
                if (printed < 20) {
                    if (printed > 0) printf(", ");
                    printf("%d", sm);
                }
                printed++;
            }
        }
        if (sm_count > 20) printf(" ... (+%d more)", sm_count - 20);
        printf("\n");
    }

    cudaFree(d_sm_ids);
    free(h_sm_ids);
    libsmctrl_set_global_mask(0);

    return 0;
}