// malloc + 多线程 pre-fault + cudaHostRegister 耗时测试
// 编译: nvcc -O2 -std=c++17 pre_malloc.cu -o pre_malloc
// 运行: ./pre_malloc

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
#include <vector>
#include <sys/mman.h>
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

static const size_t PAGE = 4096;

// ── 方式 1：手动多线程 prefault（写每页触发 page fault） ──────────────────────

static void prefault_worker(void* ptr, size_t size, int tid, int nthreads) {
    uintptr_t start = (uintptr_t)ptr + (size / nthreads) * tid;
    uintptr_t end   = (tid == nthreads - 1)
                    ? (uintptr_t)ptr + size
                    : start + (size / nthreads);
    uintptr_t p = (start + PAGE - 1) & ~(PAGE - 1);
    for (; p < end; p += PAGE)
        *(volatile char*)p = 0;
}

static void bench_prefault(size_t size, const char* label, int nthreads) {
    void* ptr = std::malloc(size);

    double t0 = now_ms();
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int i = 0; i < nthreads; ++i)
        workers.emplace_back(prefault_worker, ptr, size, i, nthreads);
    for (auto& t : workers) t.join();
    double fault_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    double reg_ms = now_ms() - t0;

    printf("  prefault    %-10s  threads: %2d  fault: %7.2f ms  register: %6.2f ms  total: %7.2f ms\n",
           label, nthreads, fault_ms, reg_ms, fault_ms + reg_ms);

    CUDA_CHECK(cudaHostUnregister(ptr));
    std::free(ptr);
}

// ── 方式 2：madvise(MADV_WILLNEED) — 内核异步预读页面 ────────────────────────
//
// MADV_WILLNEED 告诉内核"即将访问这段内存"，内核会在后台发起 readahead，
// 把匿名页提前分配并填零。调用本身很快返回，实际 fault 延迟推迟到
// cudaHostRegister 内部的内存遍历时触发（如果内核还没完成的话）。
// 通常比手动 prefault 慢，因为是单线程内核路径，但不占用用户态 CPU。

static void bench_madvise_willneed(size_t size, const char* label) {
    void* ptr = std::malloc(size);

    double t0 = now_ms();
    // madvise 是异步的，仅通知内核预取，不保证返回时页面已就绪
    madvise(ptr, size, MADV_WILLNEED);
    double adv_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    double reg_ms = now_ms() - t0;

    printf("  willneed    %-10s  threads:  -  advise: %7.2f ms  register: %6.2f ms  total: %7.2f ms\n",
           label, adv_ms, reg_ms, adv_ms + reg_ms);

    CUDA_CHECK(cudaHostUnregister(ptr));
    std::free(ptr);
}

// ── 方式 3：madvise(MADV_HUGEPAGE) + 手动 prefault ───────────────────────────
//
// MADV_HUGEPAGE 请求内核将此区域合并为 2 MB 大页（THP）。
// 大页把 page fault 次数从 N/4K 降到 N/2M（少 512 倍），
// 同时减少 TLB 压力，cudaHostRegister 内部 pin 的代价也更低。
// 需配合 prefault（写一次触发实际分配），否则大页不会立即生效。
// 注意：需要内核支持 THP（/sys/kernel/mm/transparent_hugepage/enabled != never）。

static const size_t HUGE_PAGE = 2UL * 1024 * 1024; // 2 MB

static void prefault_hugepage_worker(void* ptr, size_t size, int tid, int nthreads) {
    uintptr_t start = (uintptr_t)ptr + (size / nthreads) * tid;
    uintptr_t end   = (tid == nthreads - 1)
                    ? (uintptr_t)ptr + size
                    : start + (size / nthreads);
    // 步长改为 2 MB，每大页只写一次
    uintptr_t p = (start + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
    for (; p < end; p += HUGE_PAGE)
        *(volatile char*)p = 0;
}

static void bench_madvise_hugepage(size_t size, const char* label, int nthreads) {
    // 按 2 MB 对齐分配，确保 THP 能生效
    void* ptr = nullptr;
    posix_memalign(&ptr, HUGE_PAGE, size);

    double t0 = now_ms();
    madvise(ptr, size, MADV_HUGEPAGE);
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int i = 0; i < nthreads; ++i)
        workers.emplace_back(prefault_hugepage_worker, ptr, size, i, nthreads);
    for (auto& t : workers) t.join();
    double fault_ms = now_ms() - t0;

    t0 = now_ms();
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    double reg_ms = now_ms() - t0;

    printf("  hugepage    %-10s  threads: %2d  fault: %7.2f ms  register: %6.2f ms  total: %7.2f ms\n",
           label, nthreads, fault_ms, reg_ms, fault_ms + reg_ms);

    CUDA_CHECK(cudaHostUnregister(ptr));
    std::free(ptr);
}

int main() {
    CUDA_CHECK(cudaFree(0));

    const int HW_THREADS = (int)std::thread::hardware_concurrency();
    const size_t MB = 1024UL * 1024;
    struct { size_t size; const char* label; } cases[] = {
        {   64 * MB,   "64 MB"  },
        {  256 * MB,  "256 MB"  },
        { 1024 * MB, "1024 MB"  },
        { 4096 * MB, "4096 MB"  },
    };

    printf("硬件线程数: %d\n\n", HW_THREADS);

    for (auto& c : cases) {
        printf("\n[%s]\n", c.label);
        printf("  %-12s  %-10s  %-10s  %-24s  %-20s  %-14s\n",
               "method", "size", "threads", "fault/advise", "register", "total");
        printf("  %s\n", std::string(100, '-').c_str());

        // 方式 1：多线程 prefault（基准）
        for (int nt : {1, 4, HW_THREADS}) {
            if (nt > HW_THREADS) break;
            bench_prefault(c.size, c.label, nt);
        }

        // 方式 2：MADV_WILLNEED（内核异步，单次对比）
        bench_madvise_willneed(c.size, c.label);

        // 方式 3：MADV_HUGEPAGE + 多线程 prefault（大页减少 fault 次数）
        for (int nt : {1, 4, HW_THREADS}) {
            if (nt > HW_THREADS) break;
            bench_madvise_hugepage(c.size, c.label, nt);
        }
    }

    return 0;
}
