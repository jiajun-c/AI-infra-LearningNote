// 测试 cudaMallocHost 在不同内存大小下的分配/释放耗时
// 编译: nvcc -O2 malloc.cu -o malloc
// 运行: ./malloc

#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                \
  do {                                                                  \
    cudaError_t _e = (expr);                                            \
    if (_e != cudaSuccess) {                                            \
      fprintf(stderr, "[CUDA ERROR] %s:%d %s\n",                       \
              __FILE__, __LINE__, cudaGetErrorString(_e));              \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void bench(size_t size, const char* label) {
    void* ptr = nullptr;

    double t0 = now_ms();
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    double alloc_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaFreeHost(ptr));
    double free_ms = now_ms() - t0;

    printf("%-10s  alloc: %7.2f ms  free: %7.2f ms\n", label, alloc_ms, free_ms);
}

int main() {
    CUDA_CHECK(cudaFree(0));  // 预热，排除 CUDA context 初始化开销

    const size_t MB = 1024UL * 1024;
    struct { size_t size; const char* label; } cases[] = {
        {   1 * MB,   "1 MB"   },
        {  16 * MB,   "16 MB"  },
        {  64 * MB,   "64 MB"  },
        { 128 * MB,   "128 MB"  },
        { 256 * MB,   "256 MB" },
        {1024 * MB,  "1024 MB" },
        {2048 * MB,  "2048 MB" },
        {4096 * MB,  "4096 MB" },
    };

    printf("%-10s  %-16s  %-16s\n", "size", "alloc", "free");
    printf("%-10s  %-16s  %-16s\n", "----", "-----", "----");
    for (auto& c : cases)
        bench(c.size, c.label);

    return 0;
}