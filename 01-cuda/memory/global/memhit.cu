/**
 * 全局内存 coalescing 对比: 最好 vs 最坏
 *
 * 一次 transaction = 128 字节 (L2 cache line)
 * 一个 warp 32 线程, 每个读 4B = 128B → 理想情况 1 transaction
 *
 * 编译: nvcc -o memhit memhit.cu -arch=native -O3
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 22)  // 4M

__global__ void load_coalesced(const float *s, float *d, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}

// 所有线程都读 N 个元素, stride 导致访问分散但数据量不变
__global__ void load_strided(const float *s, float *d, int stride, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) d[i] = s[(i * (unsigned long long)stride) % n];
}

__global__ void load_random(const float *s, float *d, const int *idx, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) d[i] = s[idx[i]];
}

// AoS: 读 float4 的一个字段, 等价 stride=4 但每读只取 1/4
__global__ void load_aos(const float4 *s, float *d, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) { d[i] = ((const float*)s)[i * 4 + 0]; }
}

inline void check(cudaError_t e, const char *m) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s\n\n", p.name);

    int block = 256;
    size_t fb = N * sizeof(float);

    float *ds, *dd; int *di;
    cudaMalloc(&ds, fb);
    cudaMalloc(&dd, fb);
    cudaMalloc(&di, fb);

    int *hi = (int*)malloc(fb);
    for (int i = 0; i < N; i++) hi[i] = rand() % N;
    cudaMemcpy(di, hi, fb, cudaMemcpyHostToDevice);
    free(hi);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 200;

#define BENCH(label, kern_call) do {                                         \
    for (int w = 0; w < 10; w++) { kern_call; }                              \
    cudaDeviceSynchronize();                                                 \
    cudaEventRecord(s);                                                      \
    for (int k = 0; k < iters; k++) { kern_call; }                           \
    cudaEventRecord(e);                                                      \
    cudaDeviceSynchronize();                                                 \
    float ms; cudaEventElapsedTime(&ms, s, e);                               \
    float t = ms / iters;                                                    \
    double bw = (double)(N) * 4 / (t / 1e3) / 1e9;                          \
    printf("  %-30s  %8.4fms  %7.1f GB/s\n", label, t, bw);                 \
} while(0)

    printf("全局内存读取带宽 (全部读 %d 个 float = %.1f MB):\n\n",
           N, (double)N * 4 / 1e6);
    printf("  %-30s  %-10s  %-10s\n", "模式", "time", "GB/s");
    printf("  %-30s  %-10s  %-10s\n", "------------------------------", "----------", "----------");

    int grid = (N + block - 1) / block;

    // stride=1~32: 用同一个 ds 数组, 通过 modulo 保证不越界
    BENCH("1. coalesced (stride=1)",   (load_coalesced<<<grid,block>>>(ds,dd,N)));
    BENCH("2. stride=2",              (load_strided<<<grid,block>>>(ds,dd,2,N)));
    BENCH("3. stride=8",              (load_strided<<<grid,block>>>(ds,dd,8,N)));
    BENCH("4. stride=32",             (load_strided<<<grid,block>>>(ds,dd,32,N)));

    // AoS: ds 是 float4, 有 N 个 float4 = 4N 个 float = 16N 字节
    // 需要重新分配, 因为之前只分配了 N*4 字节
    {
        size_t aos_bytes = (size_t)N * 4 * sizeof(float);
        float *ds4;
        cudaMalloc(&ds4, aos_bytes);
        int ga = (N + block - 1) / block;
        BENCH("5. AoS (stride=4)",     (load_aos<<<ga,block>>>((float4*)ds4,dd,N)));
        cudaFree(ds4);
    }

    // random: 需要验证结果, 防止编译器优化掉
    {
        volatile float sink;  // 阻止优化
        BENCH("6. random (最坏)",      (load_random<<<grid,block>>>(ds,dd,di,N)));
    }

#undef BENCH

    printf("\n");
    printf("============================================================\n");
    printf("最坏情况 = random access, 四个层面的惩罚叠加:\n\n");
    printf("  层面 1 — Transaction 浪费:\n");
    printf("    每线程 4B 有用数据 → 触发 128B transaction\n");
    printf("    32 threads × 128B = 4KB 传输, 只用到 128B\n");
    printf("    → 带宽利用率 = 128/4096 = 3.1%%\n\n");
    printf("  层面 2 — L2 Cache miss:\n");
    printf("    无时间/空间局部性 → L2 缓存全部失效\n");
    printf("    每次访问都要走 DRAM → 延迟 200-800 cycles\n\n");
    printf("  层面 3 — TLB miss:\n");
    printf("    随机访问跨越大量内存页 → TLB miss 频繁\n");
    printf("    → 页表遍历, 额外 1-2 次内存访问\n\n");
    printf("  层面 4 — DRAM row buffer thrash:\n");
    printf("    每次访问命中不同 DRAM row\n");
    printf("    → activate + precharge 开销叠加\n");
    printf("    → DRAM 有效带宽下降 50-80%%\n\n");
    printf("  总延迟 > 1000 cycles (vs coalesced ~200 cycles)\n");
    printf("============================================================\n");
    printf("\n");
    printf("stride 对 coalescing 的影响 (L2 cache line = 128B):\n");
    printf("  stride=1:  32 threads 覆盖 128B → 1 transaction\n");
    printf("  stride=2:  32 threads 覆盖 256B → 2 transactions\n");
    printf("  stride=N:  32 threads 覆盖 N×128B → min(32, 32×N×4/128) tx\n");
    printf("  stride≥32: 32 threads 各占不同 128B → 32 transactions (最差)\n");

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(ds); cudaFree(dd); cudaFree(di);
    return 0;
}
