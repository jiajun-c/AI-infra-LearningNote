// 测试 pinned memory 各种分配方式的耗时
//
// 编译: nvcc -O2 -std=c++17 test_pin_time.cu -o test_pin_time
// 运行: ./test_pin_time

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <thread>
#include <future>

// ── 工具宏 ────────────────────────────────────────────────────
#define CUDA_CHECK(expr)                                              \
  do {                                                               \
    cudaError_t _e = (expr);                                         \
    if (_e != cudaSuccess) {                                         \
      fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                   \
              __FILE__, __LINE__, cudaGetErrorString(_e));           \
      exit(1);                                                       \
    }                                                                \
  } while (0)

// 高精度计时（ns）
static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static const size_t PAGE = 4096;

// ── 并行 pre-fault（模拟 PyTorch allocWithCudaHostRegister）─────
static void mapPages(void* ptr, size_t size, int tid, int nthreads) {
    uintptr_t start = (uintptr_t)ptr + (size * tid / nthreads);
    uintptr_t end   = start + (size / nthreads);
    if (tid == nthreads - 1)
        end = (uintptr_t)ptr + size;

    // 向上对齐到页边界
    uintptr_t p = (start + PAGE - 1) & ~(PAGE - 1);
    for (; p < end; p += PAGE)
        memset((void*)p, 0, 1);   // 触发 page fault，仅写首字节
}

static void prefault_parallel(void* ptr, size_t size, int nthreads) {
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int i = 0; i < nthreads; ++i)
        threads.emplace_back(mapPages, ptr, size, i, nthreads);
    for (auto& t : threads) t.join();
}

// ── 打印结果辅助 ──────────────────────────────────────────────
static void print_row(const char* label, double t_ms, size_t bytes) {
    double bw = (double)bytes / t_ms / 1e6;   // GB/s
    printf("  %-52s %8.2f ms  (%6.2f GB/s)\n", label, t_ms, bw);
}

// ══════════════════════════════════════════════════════════════
// 测试1: cudaMallocHost（策略A，PyTorch 默认）
// ══════════════════════════════════════════════════════════════
static void bench_cudaMallocHost(size_t size) {
    void* ptr = nullptr;

    double t0 = now_ms();
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    double alloc_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaFreeHost(ptr));
    double free_ms = now_ms() - t0;

    print_row("cudaMallocHost alloc",  alloc_ms, size);
    print_row("cudaMallocHost free",   free_ms,  size);
}

// ══════════════════════════════════════════════════════════════
// 测试2: malloc + cudaHostRegister（策略B，单线程 pre-fault）
// ══════════════════════════════════════════════════════════════
static void bench_register_single(size_t size) {
    void* ptr = nullptr;

    double t0 = now_ms();
    ptr = std::malloc(size);
    double malloc_ms = now_ms() - t0;

    t0 = now_ms();
    mapPages(ptr, size, 0, 1);        // 单线程 pre-fault
    double prefault_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    double reg_ms = now_ms() - t0;

    double total_ms = malloc_ms + prefault_ms + reg_ms;

    print_row("malloc",                         malloc_ms,   size);
    print_row("pre-fault (1 thread)",           prefault_ms, size);
    print_row("cudaHostRegister (after fault)", reg_ms,      size);
    print_row("  → total (strategy B, 1T)",    total_ms,    size);

    CUDA_CHECK(cudaHostUnregister(ptr));
    std::free(ptr);
}

// ══════════════════════════════════════════════════════════════
// 测试3: malloc + 并行 pre-fault + cudaHostRegister（策略B，多线程）
// ══════════════════════════════════════════════════════════════
static void bench_register_parallel(size_t size, int nthreads) {
    void* ptr = nullptr;

    double t0 = now_ms();
    ptr = std::malloc(size);
    double malloc_ms = now_ms() - t0;

    t0 = now_ms();
    prefault_parallel(ptr, size, nthreads);
    double prefault_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    double reg_ms = now_ms() - t0;

    double total_ms = malloc_ms + prefault_ms + reg_ms;

    char label[80];
    snprintf(label, sizeof(label), "pre-fault (%d threads)", nthreads);
    print_row("malloc",           malloc_ms,   size);
    print_row(label,              prefault_ms, size);
    print_row("cudaHostRegister", reg_ms,      size);
    snprintf(label, sizeof(label), "  → total (strategy B, %dT)", nthreads);
    print_row(label,              total_ms,    size);

    CUDA_CHECK(cudaHostUnregister(ptr));
    std::free(ptr);
}

// ══════════════════════════════════════════════════════════════
// 测试4: 无 pre-fault，直接 cudaHostRegister（对比基线）
//        cudaHostRegister 内部被迫串行触发 page fault + 持全局锁
// ══════════════════════════════════════════════════════════════
static void bench_register_no_prefault(size_t size) {
    void* ptr = std::malloc(size);

    double t0 = now_ms();
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    double reg_ms = now_ms() - t0;

    print_row("cudaHostRegister (no pre-fault, worst case)", reg_ms, size);

    CUDA_CHECK(cudaHostUnregister(ptr));
    std::free(ptr);
}

// ══════════════════════════════════════════════════════════════
// 测试5: H2D 传输速度（验证 pinned 内存真正发挥作用）
// ══════════════════════════════════════════════════════════════
static void bench_h2d_bandwidth(size_t size) {
    void* pinned = nullptr;
    void* device = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned, size));
    CUDA_CHECK(cudaMalloc(&device, size));
    memset(pinned, 1, size);

    // warm-up
    CUDA_CHECK(cudaMemcpy(device, pinned, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int ITER = 5;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITER; ++i)
        CUDA_CHECK(cudaMemcpy(device, pinned, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double t_ms = ms / ITER;

    print_row("H2D (pinned → GPU, cudaMemcpy avg)", t_ms, size);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(pinned));
    CUDA_CHECK(cudaFree(device));
}

// ══════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════
int main() {
    // 初始化 CUDA context（第一次 CUDA 调用会有 context 创建开销，排除干扰）
    CUDA_CHECK(cudaFree(0));

    const size_t MB = 1024UL * 1024;
    const size_t sizes[] = {64*MB, 256*MB, 1024*MB};
    const char*  labels[] = {"64 MB", "256 MB", "1024 MB"};
    const int    max_hw_threads = (int)std::thread::hardware_concurrency();
    const int    thread_counts[] = {2, 4, 8};

    for (int si = 0; si < 3; ++si) {
        size_t sz = sizes[si];
        printf("\n╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  size = %s\n", labels[si]);
        printf("╚══════════════════════════════════════════════════════════════╝\n");

        printf("\n── 策略A: cudaMallocHost ──────────────────────────────────────\n");
        bench_cudaMallocHost(sz);

        printf("\n── 策略B baseline: 无 pre-fault 直接 cudaHostRegister ─────────\n");
        bench_register_no_prefault(sz);

        printf("\n── 策略B: malloc + single-thread pre-fault + cudaHostRegister ─\n");
        bench_register_single(sz);

        for (int nt : thread_counts) {
            if (nt > max_hw_threads) continue;
            printf("\n── 策略B: malloc + %d-thread pre-fault + cudaHostRegister ─────\n", nt);
            bench_register_parallel(sz, nt);
        }

        printf("\n── H2D 带宽验证 ───────────────────────────────────────────────\n");
        bench_h2d_bandwidth(sz);
    }

    printf("\n硬件线程数: %d\n", max_hw_threads);
    return 0;
}
