/**
 * CUDA Core FP32 FLOPS Benchmark
 *
 * 纯 CUDA Core（非 Tensor Core）的 FP32 吞吐测试。
 * FMA 循环: acc = acc * x + y, 每次 2 FLOPs.
 *
 * 两种计时交叉验证:
 *   cudaEvent → wall-clock → GFLOPS
 *   clock64() → SM 周期   → 每线程 FLOPs/cycle 效率
 *
 * clock64() 测的是每线程 FMA 循环的 SM 周期数，不跨 SM 同步。
 * SM 频率通过 NVML 在 kernel 运行时采样（从 CPU 线程轮询）。
 *
 * 编译: nvcc -o bench_fp32 bench_fp32.cu -arch=native -O3 -lnvidia-ml
 * 运行: ./bench_fp32
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <atomic>
#include <chrono>

#define WARMUP_ITERS  20
#define BENCH_ITERS   200

// ---------------------------------------------------------------------------
// NVML
// ---------------------------------------------------------------------------
static nvmlDevice_t nvml_dev;

void nvml_init(int id) { nvmlInit(); nvmlDeviceGetHandleByIndex(id, &nvml_dev); }
void nvml_done() { nvmlShutdown(); }

// ---------------------------------------------------------------------------
// Kernel: clock64() 只测每线程自己的 FMA 周期
// ---------------------------------------------------------------------------
template <int INNER_ITERS>
__global__ void fma_bench(float *a, float *b, float *c,
                          unsigned long long *cycles, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    float acc = c[tid];
    float x   = a[tid];
    float y   = b[tid];

    unsigned long long t0 = clock64();

#pragma unroll
    for (int i = 0; i < INNER_ITERS; i++) {
        acc = acc * x + y;
    }

    unsigned long long t1 = clock64();

    c[tid]       = acc;
    cycles[tid]  = t1 - t0;
}

// ILP=4 版本: 4 条独立 FMA 依赖链, 隐藏 ~4 cycle 的 FMA 流水线延迟
// 总 FLOPs 不变 (INNER_ITERS * 2), 但 4 条链可以同时发射
template <int INNER_ITERS>
__global__ void fma_bench_ilp4(float *a, float *b, float *c,
                                unsigned long long *cycles, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    float x = a[tid], y = b[tid];
    float acc0 = c[tid], acc1 = c[tid], acc2 = c[tid], acc3 = c[tid];

    unsigned long long t0 = clock64();

    // 4 条链各自展开, 互不依赖
#pragma unroll
    for (int i = 0; i < INNER_ITERS / 4; i++) {
        acc0 = acc0 * x + y;
        acc1 = acc1 * x + y;
        acc2 = acc2 * x + y;
        acc3 = acc3 * x + y;
    }

    unsigned long long t1 = clock64();

    c[tid]       = acc0 + acc1 + acc2 + acc3;
    cycles[tid]  = t1 - t0;
}

inline void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// 轮询 NVML 采样 SM 频率 (从 CPU 线程, kernel 运行期间)
// ---------------------------------------------------------------------------
std::atomic<bool> polling{false};
std::atomic<unsigned int> max_sm_clock{0};

void nvml_poll_loop() {
    while (polling.load()) {
        unsigned int clk = 0;
        nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_SM, &clk);
        if (clk > max_sm_clock.load()) max_sm_clock.store(clk);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

static float g_peak_fpc = 0;  // 在 main 中初始化
static int   g_sm_count = 0;

// ---------------------------------------------------------------------------
// Benchmark (单链)
// ---------------------------------------------------------------------------
template <int INNER_ITERS>
void bench_fma(const char *label, int N) {
    float *a, *b, *c;
    unsigned long long *d_cycles, *h_cycles;
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    size_t bf = N * sizeof(float);
    size_t bc = N * sizeof(unsigned long long);
    check(cudaMalloc(&a, bf), "malloc a");
    check(cudaMalloc(&b, bf), "malloc b");
    check(cudaMalloc(&c, bf), "malloc c");
    check(cudaMalloc(&d_cycles, bc), "malloc cycles");
    h_cycles = (unsigned long long*)malloc(bc);

    float *ha = (float*)malloc(bf), *hb = (float*)malloc(bf), *hc = (float*)malloc(bf);
    for (int i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; hc[i] = 0.0f; }
    check(cudaMemcpy(a, ha, bf, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(b, hb, bf, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(c, hc, bf, cudaMemcpyHostToDevice), "H2D");
    free(ha); free(hb); free(hc);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // === warmup ===
    for (int i = 0; i < WARMUP_ITERS; i++)
        fma_bench<INNER_ITERS><<<gridSize, blockSize>>>(a, b, c, d_cycles, N);
    cudaDeviceSynchronize();

    // === 启动 NVML 轮询线程 + benchmark ===
    polling.store(true);
    max_sm_clock.store(0);
    std::thread poller(nvml_poll_loop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        fma_bench<INNER_ITERS><<<gridSize, blockSize>>>(a, b, c, d_cycles, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    polling.store(false);
    poller.join();
    unsigned int sm_clock_mhz = max_sm_clock.load();

    float total_ms, time_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    time_ms = total_ms / BENCH_ITERS;

    // 读回 cycle 数据
    check(cudaMemcpy(h_cycles, d_cycles, bc, cudaMemcpyDeviceToHost), "D2H");

    // 统计每线程周期数
    unsigned long long sum_c = 0, max_c = 0, min_c = ~0ULL;
    for (int i = 0; i < N; i++) {
        sum_c += h_cycles[i];
        if (h_cycles[i] > max_c) max_c = h_cycles[i];
        if (h_cycles[i] < min_c) min_c = h_cycles[i];
    }
    double avg_cycles = (double)sum_c / N;

    // --- 从 min_cyc (最快线程的 t1-t0) 计算单线程 FLOPs/cycle ---
    // 这反映单条依赖链的指令级效率, 不受 occupancy/线程等待影响
    double thread_flops = (double)(INNER_ITERS * 2);
    double thread_fpc   = thread_flops / (double)min_c;  // FLOPs per cycle per thread

    // --- FLOPs ---
    double flops_per_call = (double)N * INNER_ITERS * 2;

    float peak_per_call = (float)g_sm_count * 128 * (sm_clock_mhz / 1000.0f) * 2;
    double gflops_event = flops_per_call / (time_ms / 1000.0) / 1e9;
    double util_pct = 100.0 * gflops_event / peak_per_call;

    double total_cycles_gpu = time_ms / 1000.0 * sm_clock_mhz * 1e6;
    double flops_per_cycle  = flops_per_call / total_cycles_gpu;

    // 带宽 & AI
    double bytes_per_call = (double)N * 2 * 4 * 2;
    double bw_gb_s = bytes_per_call / (time_ms / 1000.0) / 1e9;
    double ai = flops_per_call / bytes_per_call;

    printf("  %-18s %7.4fms  %8.1f GF  %5.1f%%  %4uMHz  "
           "%4.0fcyc %5.2ftFPC  %8.1f FPC  %6.1f GB/s  AI=%.0f\n",
           label, time_ms,
           gflops_event, util_pct,
           sm_clock_mhz, (double)min_c, thread_fpc,
           flops_per_cycle, bw_gb_s, ai);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaFree(d_cycles);
    free(h_cycles);
}

// ILP=4 版本
template <int INNER_ITERS>
void bench_fma_ilp4(const char *label, int N) {
    float *a, *b, *c;
    unsigned long long *d_cycles, *h_cycles;
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    size_t bf = N * sizeof(float);
    size_t bc = N * sizeof(unsigned long long);
    check(cudaMalloc(&a, bf), "malloc a");
    check(cudaMalloc(&b, bf), "malloc b");
    check(cudaMalloc(&c, bf), "malloc c");
    check(cudaMalloc(&d_cycles, bc), "malloc cycles");
    h_cycles = (unsigned long long*)malloc(bc);

    float *ha = (float*)malloc(bf), *hb = (float*)malloc(bf), *hc = (float*)malloc(bf);
    for (int i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; hc[i] = 0.0f; }
    check(cudaMemcpy(a, ha, bf, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(b, hb, bf, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(c, hc, bf, cudaMemcpyHostToDevice), "H2D");
    free(ha); free(hb); free(hc);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++)
        fma_bench_ilp4<INNER_ITERS><<<gridSize, blockSize>>>(a, b, c, d_cycles, N);
    cudaDeviceSynchronize();

    polling.store(true); max_sm_clock.store(0);
    std::thread poller(nvml_poll_loop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        fma_bench_ilp4<INNER_ITERS><<<gridSize, blockSize>>>(a, b, c, d_cycles, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    polling.store(false); poller.join();
    unsigned int sm_clock_mhz = max_sm_clock.load();

    float total_ms, time_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    time_ms = total_ms / BENCH_ITERS;

    check(cudaMemcpy(h_cycles, d_cycles, bc, cudaMemcpyDeviceToHost), "D2H");

    unsigned long long sum_c = 0, max_c = 0, min_c = ~0ULL;
    for (int i = 0; i < N; i++) {
        sum_c += h_cycles[i];
        if (h_cycles[i] > max_c) max_c = h_cycles[i];
        if (h_cycles[i] < min_c) min_c = h_cycles[i];
    }
    double avg_cycles = (double)sum_c / N;

    // 从 min_cyc (最快线程的 t1-t0) 算单线程 FLOPs/cycle
    double thread_flops = (double)(INNER_ITERS * 2);
    double thread_fpc   = thread_flops / (double)min_c;

    double flops_per_call = (double)N * INNER_ITERS * 2;

    float peak_per_call = (float)g_sm_count * 128 * (sm_clock_mhz / 1000.0f) * 2;
    double gflops_event = flops_per_call / (time_ms / 1000.0) / 1e9;
    double util_pct = 100.0 * gflops_event / peak_per_call;

    double total_cycles_gpu = time_ms / 1000.0 * sm_clock_mhz * 1e6;
    double flops_per_cycle  = flops_per_call / total_cycles_gpu;

    double bytes_per_call = (double)N * 2 * 4 * 2;
    double bw_gb_s = bytes_per_call / (time_ms / 1000.0) / 1e9;
    double ai = flops_per_call / bytes_per_call;

    printf("  %-18s %7.4fms  %8.1f GF  %5.1f%%  %4uMHz  "
           "%4.0fcyc %5.2ftFPC  %8.1f FPC  %6.1f GB/s  AI=%.0f\n",
           label, time_ms,
           gflops_event, util_pct,
           sm_clock_mhz, (double)min_c, thread_fpc,
           flops_per_cycle, bw_gb_s, ai);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaFree(d_cycles);
    free(h_cycles);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    g_sm_count = sm_count;

    int base_khz = 0;
    cudaDeviceGetAttribute(&base_khz, cudaDevAttrClockRate, 0);

    nvml_init(0);
    unsigned int max_p0 = 0;
    nvmlDeviceGetMaxClockInfo(nvml_dev, NVML_CLOCK_SM, &max_p0);

    const int N = 2 << 22;
    int block = 256, grid = (N + block - 1) / block;
    float *a, *b, *c; unsigned long long *cyc;
    cudaMalloc(&a, N*4); cudaMalloc(&b, N*4); cudaMalloc(&c, N*4); cudaMalloc(&cyc, N*8);

    for (int i = 0; i < 50; i++)
        fma_bench<512><<<grid, block>>>(a, b, c, cyc, N);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    polling.store(true); max_sm_clock.store(0);
    std::thread poller(nvml_poll_loop);

    cudaEventRecord(s);
    for (int i = 0; i < 100; i++)
        fma_bench<512><<<grid, block>>>(a, b, c, cyc, N);
    cudaEventRecord(e);
    cudaDeviceSynchronize();

    polling.store(false); poller.join();
    unsigned int sm_clock_mhz = max_sm_clock.load();

    float ms; cudaEventElapsedTime(&ms, s, e);
    float peak_gflops = sm_count * 128 * (sm_clock_mhz / 1000.0f) * 2;

    printf("============================================================\n");
    printf("GPU: %s  |  CC: %d.%d  |  SM: %d  |  Cores: %d\n",
           prop.name, prop.major, prop.minor, sm_count, sm_count * 128);
    printf("\nSM Clock:\n");
    printf("  base  (cudaDevAttrClockRate):    %d MHz\n", base_khz/1000);
    printf("  max   (nvmlDeviceGetMaxClockInfo): %u MHz (P0)\n", max_p0);
    printf("  load  (NVML poll during kernel):   %u MHz\n", sm_clock_mhz);
    printf("\n理论峰值 FP32 (@load clock): %.0f GFLOPS\n", peak_gflops);
    printf("============================================================\n");

    cudaEventDestroy(s); cudaEventDestroy(e);

    g_peak_fpc = (float)(sm_count * 128 * 2);

    printf("\n--- CUDA Core FP32 FMA Benchmark ---\n");
    printf("  FPC = FLOPs/cycle (全 GPU), 峰值 = %.0f\n", g_peak_fpc);
    printf("  util%% 按每行采样的 MHz 单独计算 (不共用 warmup 的 clock)\n");
    printf("  %-18s %-9s %-10s %-7s %-7s %-7s %-7s %-10s %-9s %s\n",
           "config", "time", "GF(event)", "util%",
           "MHz", "min_cyc", "tFPC", "FPC", "GB/s", "AI");
    printf("  %-18s %-9s %-10s %-7s %-7s %-7s %-7s %-10s %-9s %s\n",
           "------------------", "---------", "----------",
           "------", "-------", "------", "-------", "----------", "--------", "---");

    bench_fma<1>   ("inner_iters=1",    N);
    bench_fma<8>   ("inner_iters=8",    N);
    bench_fma<32>  ("inner_iters=32",   N);
    bench_fma<128> ("inner_iters=128",  N);
    bench_fma<512> ("inner_iters=512",  N);
    bench_fma<2048>("inner_iters=2048", N);

    printf("\n--- ILP=4 (4条独立FMA链) ---\n");
    bench_fma_ilp4<1>   ("ILP4 inner=1",    N);
    bench_fma_ilp4<8>   ("ILP4 inner=8",    N);
    bench_fma_ilp4<32>  ("ILP4 inner=32",   N);
    bench_fma_ilp4<128> ("ILP4 inner=128",  N);
    bench_fma_ilp4<512> ("ILP4 inner=512",  N);
    bench_fma_ilp4<2048>("ILP4 inner=2048", N);

    printf("\n============================================================\n");
    printf("列说明:\n");
    printf("  GF(event): cudaEvent wall-clock → GFLOPS (最可靠)\n");
    printf("  util%%:    每行用本行采样的 MHz 单独算利用率\n");
    printf("  FPC:       FLOPs/cycle (全 GPU), 峰值 %.0f\n", g_peak_fpc);
    printf("  min_cyc:   最快线程的 clock64() 周期 (t1-t0), ≈纯执行\n");
    printf("  tFPC:      单线程 FLOPs/cycle = INNER×2 / min_cyc\n");
    printf("             反映指令级效率: 单链≈0.25, ILP4≈1.5+\n");
    printf("  MHz:       NVML 轮询采样的 SM 频率 (10ms 间隔)\n");
    printf("\n");
    printf("ILP 对比:\n");
    printf("  单链: acc = acc*x+y   → 1条依赖链, FMA延迟~4cyc\n");
    printf("  ILP4: 4条独立链        → 4倍并行, 隐藏延迟\n");
    printf("  关键看 min_cyc: ILP 应减少到 1/4, FPC 应提升\n");

    cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(cyc);
    nvml_done();
    return 0;
}
