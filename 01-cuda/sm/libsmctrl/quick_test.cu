/**
 * 快速测试：展示如何使用 libsmctrl 限制 kernel 的 SM 使用量
 *
 * 编译：
 *   nvcc -o quick_test quick_test.cu -L/volume/code/jjcheng/libsmctrl -lsmctrl -lcuda -I/volume/code/jjcheng/libsmctrl
 *
 * 运行：
 *   ./quick_test        # 不限制
 *   ./quick_test 64     # 限制 64 个 SM
 *   ./quick_test 32     # 限制 32 个 SM
 */

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

// 简单的 element-wise kernel
__global__ void elementwise_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 2.0f * x[idx] + 1.0f;
    }
}

int get_sm_count() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

int main(int argc, char** argv) {
    int sm_limit = -1;
    if (argc > 1) {
        sm_limit = atoi(argv[1]);
    }

    // 获取设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int actual_sm = prop.multiProcessorCount;

    printf("GPU: %s (SM 数量：%d)\n", prop.name, actual_sm);

    // 设置 SM 限制
    if (sm_limit > 0 && sm_limit < actual_sm) {
        // 掩码位=1 表示禁用该 SM
        // 启用前 sm_limit 个 SM，禁用其他的
        uint64_t mask = ~((uint64_t)1 << sm_limit) + 1;
        libsmctrl_set_next_mask(mask);
        printf("限制 SM 数量：%d (掩码：0x%016lx)\n", sm_limit, mask);
    } else {
        printf("不限制 SM 数量（使用全部 %d 个）\n", actual_sm);
    }

    // 准备数据
    int array_size = 50000000;  // 5000 万
    size_t bytes = array_size * sizeof(float);
    float *x, *y;
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));

    // 初始化
    float *host_x = (float*)malloc(bytes);
    for (int i = 0; i < array_size; i++) host_x[i] = i * 0.01f;
    CUDA_CHECK(cudaMemcpy(x, host_x, bytes, cudaMemcpyHostToDevice));
    free(host_x);

    // 配置 kernel
    int threads = 256;
    int blocks = (array_size + threads - 1) / threads;

    // warmup
    elementwise_kernel<<<blocks, threads>>>(x, y, array_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        elementwise_kernel<<<blocks, threads>>>(x, y, array_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

    printf("平均耗时：%.3f ms (%.3f ms/iter)\n", elapsed, elapsed / iterations);

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));

    printf("\n完成。\n");
    return 0;
}
