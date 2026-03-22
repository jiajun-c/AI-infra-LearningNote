#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

// ============================================================================
// 可替换的计算函数（__device__）
//
// 将计算逻辑集中到一个 device 函数中，便于：
//   1. 统一修改计算内容，保证三种 kernel 做同样的计算
//   2. 通过增减计算量来观察流水线/Warp特化对 compute-bound 场景的收益
//
// 当前实现：多次迭代的 math-heavy 计算（sinf/expf），使计算耗时远超访存，
// 从而让流水线的 load-compute 重叠和 Warp 特化的并行优势得以体现。
// 如需回退为轻量计算，将循环次数改为 0 或直接 return val * 2.0f 即可。
// ============================================================================
static constexpr int COMPUTE_ITERS = 1;  // 计算迭代次数，增大则更 compute-bound

__device__ __forceinline__ float compute_func(float val) {
    float result = val;
    #pragma unroll 1
    for (int iter = 0; iter < COMPUTE_ITERS; ++iter) {
        result = sinf(result) * expf(-result * 0.01f) + val;
    }
    return result;
}

// CPU 端参考实现，与 device 函数保持一致
inline float compute_func_host(float val) {
    float result = val;
    for (int iter = 0; iter < COMPUTE_ITERS; ++iter) {
        result = sinf(result) * expf(-result * 0.01f) + val;
    }
    return result;
}

// ============================================================================
// Kernel：朴素版本（无流水线，无 Shared Memory）
//
// 最简单的 baseline：每个线程直接从 Global Memory 读取，计算后写回。
// 无任何优化手段，用于与流水线版本和 Warp 特化版本做性能对比。
// ============================================================================
__global__ void naive_vec_x2(float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        out[i] = compute_func(in[i]);
    }
}

// ============================================================================
// 双缓冲流水线示例：向量逐元素计算（基于 Hopper mbarrier）
//
// 核心思想：
//   使用 2 个 shared memory buffer（stage），Producer 和 Consumer 交替工作：
//   - Producer：从 Global Memory 加载数据到 Shared Memory
//   - Consumer：从 Shared Memory 读取数据，计算后写回 Global Memory
//   通过 mbarrier 实现 Producer/Consumer 之间的同步，避免数据竞争。
//
// 流水线时序（双缓冲）：
//   Stage 0: [Load tile 0] [Compute tile 0] [Load tile 2] [Compute tile 2] ...
//   Stage 1:               [Load tile 1]    [Compute tile 1] [Load tile 3] ...
// ============================================================================

// ----- 常量定义 -----
static constexpr int PIPE_STAGES    = 2;                          // 流水线级数（双缓冲）
static constexpr int BLOCK_SIZE     = 128;                        // 每个 block 的线程数
static constexpr int TILE_SIZE      = BLOCK_SIZE;                 // 每个 tile 处理的元素数（= 线程数，每线程搬运 1 个 float）
static constexpr int SMEM_PER_STAGE = TILE_SIZE * sizeof(float);  // 每个 stage 占用的 shared memory 大小

// ----- Warp 特化常量 -----
static constexpr int PRODUCER_THREADS = 32;   // Producer warp 线程数（Warp 0）
static constexpr int CONSUMER_THREADS = 96;   // Consumer warp 线程数（Warps 1-3）

// ----- Shared Memory 布局 -----
struct SharedStorage {
    alignas(128) float buf[PIPE_STAGES][TILE_SIZE];  // 双缓冲数据区，128 字节对齐以满足 cp.async 要求
    uint64_t producer_mbar[PIPE_STAGES];             // Producer barrier：通知 Consumer "数据已就绪"
    uint64_t consumer_mbar[PIPE_STAGES];             // Consumer barrier：通知 Producer "buffer 已释放，可覆写"

};

// ============================================================================
// Kernel：双缓冲流水线向量 ×2
//
// 参数：
//   in  - 输入数组（Global Memory）
//   out - 输出数组（Global Memory）
//   N   - 数组元素总数
// ============================================================================
__global__ void pipe_vec_v2(float* in, float* out, int N) {
    extern __shared__ char shared_memory[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    int tid = threadIdx.x;

    // Barrier 类型定义（Hopper 架构的硬件 mbarrier）
    using ProducerBarType = cutlass::arch::ClusterBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    uint64_t* producer_mbar = smem.producer_mbar;
    uint64_t* consumer_mbar = smem.consumer_mbar;

    // ========================================================================
    // Step 1：初始化 mbarrier
    //   - 每个 barrier 的 arrive 计数设为 BLOCK_SIZE（所有线程都需要 arrive）
    //   - 只需 tid==0 执行 init，随后 __syncthreads 确保所有线程可见
    // ========================================================================
    if (tid == 0) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe], BLOCK_SIZE);
            ConsumerBarType::init(&consumer_mbar[pipe], BLOCK_SIZE);
        }
    }
    __syncthreads();

    // 计算当前 block 需要处理的 tile 总数（grid-stride 循环分配）
    int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int my_tile_count = 0;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        my_tile_count++;
    }

    // PipelineState 维护当前 stage 索引和 phase（用于 mbarrier 的 flip-flop 机制）
    auto write_state = cutlass::PipelineState<PIPE_STAGES>();  // Producer 写入状态
    auto read_state  = cutlass::PipelineState<PIPE_STAGES>();  // Consumer 读取状态

    int k_tile_count = my_tile_count;  // 剩余待加载的 tile 数
    int load_tile    = blockIdx.x;     // 下一个要加载的 tile 全局索引
    int store_tile   = blockIdx.x;     // 下一个要写回的 tile 全局索引

    // ========================================================================
    // Step 2：Prefetch 阶段 —— 预填充所有 pipeline stage
    //   在主循环开始前，先把前 PIPE_STAGES 个 tile 从 Global Memory
    //   加载到 Shared Memory 的各个 stage 中，以便主循环一开始就有数据可消费。
    // ========================================================================
    CUTE_UNROLL
    for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
        if (k_tile_count > 0) {
            int global_offset = load_tile * TILE_SIZE;

            // 所有线程协作加载：每个线程搬运 1 个 float 到 shared memory
            if (global_offset + tid < N) {
                smem.buf[pipe][tid] = in[global_offset + tid];
            } else {
                smem.buf[pipe][tid] = 0.0f;  // 越界部分填零
            }

            // 所有线程 arrive，通知 Consumer "这个 stage 的数据已就绪"
            ProducerBarType::arrive(&producer_mbar[pipe]);

            // __syncthreads 确保 128 个线程都已将数据写入 SMEM
            __syncthreads();

            --k_tile_count;
            load_tile += gridDim.x;
        }
    }
    __syncthreads();  // 确保 prefetch 阶段完全结束

    // 重置计数器，主循环需要从头消费所有 tile
    k_tile_count = my_tile_count;

    // ========================================================================
    // Step 3：流水线主循环（Producer-Consumer 交替执行）
    //
    //   循环条件 k_tile_count > -PIPE_STAGES：
    //   即使没有新 tile 要加载（k_tile_count <= 0），Consumer 仍需消费
    //   prefetch 阶段遗留在 buffer 中的数据，因此需要额外迭代 PIPE_STAGES 次。
    // ========================================================================
    CUTE_NO_UNROLL
    while (k_tile_count > -PIPE_STAGES)
    {
        // ---- Consumer 端：消费当前 stage 的数据 ----
        int read_pipe = read_state.index();  // 当前要读取的 stage 索引（0 或 1）

        // 等待 Producer 完成当前 stage 的数据加载
        // mbarrier::wait 会阻塞直到 arrive 计数达标且 phase 匹配
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        // 执行计算：读取 SMEM → compute_func → 写回 Global Memory
        float val = compute_func(smem.buf[read_pipe][tid]);
        int global_offset = store_tile * TILE_SIZE;
        if (global_offset + tid < N) {
            out[global_offset + tid] = val;
        }
        store_tile += gridDim.x;

        // __syncthreads 确保所有线程都已完成计算和写回
        __syncthreads();

        // 通知 Producer "当前 stage 的 buffer 已消费完毕，可以覆写"
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;  // 推进 Consumer 状态（切换到下一个 stage）

        // ---- Producer 端：加载新 tile 到已释放的 stage ----
        if (k_tile_count > 0) {
            int write_pipe = write_state.index();  // 当前要写入的 stage 索引

            // 等待 Consumer 释放当前 stage 的 buffer
            ConsumerBarType::wait(&consumer_mbar[write_pipe], write_state.phase());

            // 所有线程协作将新 tile 从 Global Memory 搬运到 SMEM
            int load_global_offset = load_tile * TILE_SIZE;
            if (load_global_offset + tid < N) {
                smem.buf[write_pipe][tid] = in[load_global_offset + tid];
            } else {
                smem.buf[write_pipe][tid] = 0.0f;
            }

            // __syncthreads 确保新数据全部写入 SMEM
            __syncthreads();

            // 通知 Consumer "新数据已就绪"
            ProducerBarType::arrive(&producer_mbar[write_pipe]);
            ++write_state;  // 推进 Producer 状态（切换到下一个 stage）
            load_tile += gridDim.x;
        }

        --k_tile_count;
    }
}

// ============================================================================
// Kernel：Warp 特化流水线向量 ×2
//
// 与 pipe_vec_v2 的区别：
//   - pipe_vec_v2：所有 128 个线程同时充当 Producer 和 Consumer（串行）
//   - pipe_vec_warp_specialized：Warp 0 (32线程) 专职 Producer，
//     Warps 1-3 (96线程) 专职 Consumer，两者通过 mbarrier 并行执行
//
// Warp 角色分配：
//   Warp 0 (tid 0-31)   → Producer：GMEM → SMEM 数据搬运
//   Warp 1-3 (tid 32-127) → Consumer：SMEM 读取 → ×2 计算 → GMEM 写回
//
// 同步机制：
//   producer_mbar (arrive=32)：Producer 通知 Consumer "数据已就绪"
//   consumer_mbar (arrive=96)：Consumer 通知 Producer "buffer 可覆写"
//   主循环中不使用 __syncthreads，完全依赖 mbarrier 实现无锁流水线
//
// 流水线时序（双缓冲，Producer/Consumer 并行）：
//   Producer:  [Prefetch S0] [Prefetch S1] [Wait cons→Load S0] [Wait cons→Load S1] ...
//   Consumer:                              [Wait prod→Compute S0] [Wait prod→Compute S1] ...
// ============================================================================

__global__ void pipe_vec_warp_specialized(float* in, float* out, int N) {
    extern __shared__ char shared_memory[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    int tid = threadIdx.x;
    int warp_idx = tid / 32;               // warp 索引：0, 1, 2, 3
    bool is_producer = (warp_idx == 0);    // Warp 0 = Producer

    // Barrier 类型定义（Hopper 架构的硬件 mbarrier）
    using ProducerBarType = cutlass::arch::ClusterBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    uint64_t* producer_mbar = smem.producer_mbar;
    uint64_t* consumer_mbar = smem.consumer_mbar;

    // ========================================================================
    // Step 1：初始化 mbarrier
    //   - producer_mbar arrive 计数 = 32（仅 Producer warp 的线程 arrive）
    //   - consumer_mbar arrive 计数 = 96（仅 Consumer warp 的线程 arrive）
    //   - 由 tid==0 初始化，__syncthreads 保证所有线程可见
    //   - 这是整个 kernel 中唯一一次 __syncthreads
    // ========================================================================
    if (tid == 0) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe], PRODUCER_THREADS);  // 32
            ConsumerBarType::init(&consumer_mbar[pipe], CONSUMER_THREADS);  // 96
        }
    }
    __syncthreads();  // 初始化后的唯一一次全 block 同步，此后完全依赖 mbarrier

    // 计算当前 block 需要处理的 tile 总数（grid-stride 循环分配）
    int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int my_tile_count = 0;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        my_tile_count++;
    }

    // ====================================================================
    // Step 2：Warp 分叉 —— Producer 和 Consumer 走不同的代码路径
    //   此后不再使用 __syncthreads，完全依赖 mbarrier 同步
    // ====================================================================
    if (is_producer) {
        // ================================================================
        // Producer 路径（Warp 0，32 线程）
        //   职责：从 Global Memory 搬运数据到 Shared Memory
        //   每个线程搬运 TILE_SIZE / 32 = 4 个 float（stride-32 访问模式）
        // ================================================================
        int producer_lane = tid;  // 0..31
        auto write_state = cutlass::PipelineState<PIPE_STAGES>();
        int k_tile_count = my_tile_count;
        int load_tile = blockIdx.x;

        // ---- Prefetch 阶段：预填充所有流水线 stage ----
        //   在 Consumer 开始消费之前，将前 PIPE_STAGES 个 tile 加载到 SMEM
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            if (k_tile_count > 0) {
                int global_offset = load_tile * TILE_SIZE;

                // 32 个线程协作加载 128 个 float，每线程 4 个
                CUTE_UNROLL
                for (int i = 0; i < TILE_SIZE / PRODUCER_THREADS; ++i) {
                    int elem_idx = producer_lane + i * PRODUCER_THREADS;
                    if (global_offset + elem_idx < N) {
                        smem.buf[pipe][elem_idx] = in[global_offset + elem_idx];
                    } else {
                        smem.buf[pipe][elem_idx] = 0.0f;
                    }
                }

                // 所有 32 个 Producer 线程 arrive，通知 Consumer "数据已就绪"
                ProducerBarType::arrive(&producer_mbar[pipe]);
                --k_tile_count;
                load_tile += gridDim.x;
            }
        }

        // Prefetch 阶段不推进 write_state：
        //   prefetch 只做了 arrive producer_mbar（通知 Consumer 数据就绪），
        //   但 Consumer 尚未 arrive consumer_mbar（尚未释放 buffer）。
        //   如果在 prefetch 中推进 write_state，主循环中 Producer 等待 consumer_mbar
        //   时的 phase 会与 consumer_mbar 的实际 phase 不同步，导致死锁。
        //   因此 write_state 保持初始值 {index=0, phase=0}，与 consumer_mbar 的
        //   初始 phase=0 对齐，确保主循环第一次 wait 能在 Consumer 消费完 stage 0 后返回。

        // ---- 主循环：等待 Consumer 释放 buffer → 加载新 tile → 通知 Consumer ----
        CUTE_NO_UNROLL
        while (k_tile_count > 0) {
            int write_pipe = write_state.index();

            // 等待 Consumer 完成对当前 stage 的消费（consumer_mbar 的 96 个线程都已 arrive）
            ConsumerBarType::wait(&consumer_mbar[write_pipe], write_state.phase());

            // 32 个线程协作将新 tile 从 GMEM 搬运到 SMEM
            int global_offset = load_tile * TILE_SIZE;
            CUTE_UNROLL
            for (int i = 0; i < TILE_SIZE / PRODUCER_THREADS; ++i) {
                int elem_idx = producer_lane + i * PRODUCER_THREADS;
                if (global_offset + elem_idx < N) {
                    smem.buf[write_pipe][elem_idx] = in[global_offset + elem_idx];
                } else {
                    smem.buf[write_pipe][elem_idx] = 0.0f;
                }
            }

            // 通知 Consumer "新数据已就绪"
            ProducerBarType::arrive(&producer_mbar[write_pipe]);
            ++write_state;
            --k_tile_count;
            load_tile += gridDim.x;
        }

    } else {
        // ================================================================
        // Consumer 路径（Warps 1-3，96 线程）
        //   职责：从 Shared Memory 读取数据，计算 ×2，写回 Global Memory
        //   96 个线程以 stride-96 方式处理 128 个元素
        // ================================================================
        int consumer_tid = tid - PRODUCER_THREADS;  // 本地索引 0..95
        auto read_state = cutlass::PipelineState<PIPE_STAGES>();
        int k_tile_count = my_tile_count;
        int store_tile = blockIdx.x;

        // ---- 主循环：等待 Producer 填充数据 → 计算 ×2 → 写回 GMEM → 释放 buffer ----
        CUTE_NO_UNROLL
        while (k_tile_count > 0) {
            int read_pipe = read_state.index();

            // 等待 Producer 完成当前 stage 的数据加载
            // mbarrier::wait 阻塞直到 32 个 Producer 线程都已 arrive 且 phase 匹配
            ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

            // 96 个线程以 stride 方式处理 128 个元素，部分线程处理 2 个元素
            int global_offset = store_tile * TILE_SIZE;
            for (int i = consumer_tid; i < TILE_SIZE; i += CONSUMER_THREADS) {
                float val = compute_func(smem.buf[read_pipe][i]);
                if (global_offset + i < N) {
                    out[global_offset + i] = val;
                }
            }

            // 所有 96 个 Consumer 线程 arrive，通知 Producer "buffer 已消费完毕，可覆写"
            ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
            ++read_state;
            --k_tile_count;
            store_tile += gridDim.x;
        }
    }
}

// ============================================================================
// 验证结果正确性
// ============================================================================
void verify_result(const float* h_out, const float* h_ref, int N, const char* label) {
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(h_out[i] - h_ref[i]);
        if (diff > 1e-5f) {
            if (error_count < 5) {
                printf("  [%s] Mismatch at [%d]: got %.4f, expected %.4f\n", label, i, h_out[i], h_ref[i]);
            }
            error_count++;
        }
        max_error = std::fmax(max_error, diff);
    }
    if (error_count == 0) {
        printf("[%s] Result: PASSED (max error: %.6f)\n", label, max_error);
    } else {
        printf("[%s] Result: FAILED (%d errors, max error: %.6f)\n", label, error_count, max_error);
    }
}

// ============================================================================
// Main：正确性验证 + 性能对比（朴素 vs 统一线程流水线 vs Warp 特化流水线）
// ============================================================================
int main(int argc, char** argv) {
    // 默认处理 1M 个 float，也可通过命令行参数指定
    int N = 1024 * 1024;
    if (argc >= 2) N = atoi(argv[1]);

    printf("Vector x2 — 朴素 vs 统一线程流水线 vs Warp 特化流水线\n");
    printf("N = %d (%.2f MB)\n\n", N, N * sizeof(float) / (1024.0f * 1024.0f));

    // --- Host 端数据准备 ---
    std::vector<float> h_in(N), h_ref(N);
    std::vector<float> h_out(N, 0.0f);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(rand() % 100) / 10.0f;
        h_ref[i] = compute_func_host(h_in[i]);  // CPU 参考结果
    }

    // --- Device 端内存分配与数据拷贝 ---
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // --- 启动参数计算 ---
    int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks = std::min(total_tiles, 256);  // 最多 256 个 block
    int smem_size = sizeof(SharedStorage);

    // --- CUDA Event 用于精确计时 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int warmup_iters = 10;
    const int bench_iters  = 100;
    float elapsed_ms = 0.0f;

    // 数据量（读 + 写）用于计算带宽
    double data_bytes = 2.0 * N * sizeof(float);  // 读入 + 写出

    // ====================================================================
    // 测试 0：朴素版本 (naive_vec_x2)
    // ====================================================================
    printf("=== 测试 0：朴素版本 (naive_vec_x2) ===\n");
    printf("Blocks: %d, Threads: %d, 无 Shared Memory\n", num_blocks, BLOCK_SIZE);

    // 正确性验证
    cudaMemset(d_out, 0, N * sizeof(float));
    naive_vec_x2<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_out.data(), h_ref.data(), N, "朴素版本");

    // 性能测试
    for (int i = 0; i < warmup_iters; i++) {
        naive_vec_x2<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++) {
        naive_vec_x2<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float naive_avg_ms = elapsed_ms / bench_iters;
    double naive_bw = data_bytes / (naive_avg_ms * 1e-3) / 1e9;  // GB/s
    printf("[朴素版本] Avg Time: %.4f ms, Bandwidth: %.1f GB/s\n", naive_avg_ms, naive_bw);

    // ====================================================================
    // 测试 1：统一线程流水线 (pipe_vec_v2)
    // ====================================================================
    printf("\n=== 测试 1：统一线程流水线 (pipe_vec_v2) ===\n");
    printf("Blocks: %d, Threads: %d, Smem: %d bytes, Tiles: %d\n",
           num_blocks, BLOCK_SIZE, smem_size, total_tiles);

    // 正确性验证
    cudaMemset(d_out, 0, N * sizeof(float));
    pipe_vec_v2<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_in, d_out, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_out.data(), h_ref.data(), N, "统一线程");

    // 性能测试
    for (int i = 0; i < warmup_iters; i++) {
        pipe_vec_v2<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++) {
        pipe_vec_v2<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float pipe_avg_ms = elapsed_ms / bench_iters;
    double pipe_bw = data_bytes / (pipe_avg_ms * 1e-3) / 1e9;
    printf("[统一线程] Avg Time: %.4f ms, Bandwidth: %.1f GB/s\n", pipe_avg_ms, pipe_bw);

    // ====================================================================
    // 测试 2：Warp 特化流水线 (pipe_vec_warp_specialized)
    // ====================================================================
    printf("\n=== 测试 2：Warp 特化流水线 (pipe_vec_warp_specialized) ===\n");
    printf("Blocks: %d, Threads: %d, Smem: %d bytes, Tiles: %d\n",
           num_blocks, BLOCK_SIZE, smem_size, total_tiles);
    printf("Producer: Warp 0 (%d threads), Consumer: Warps 1-3 (%d threads)\n",
           PRODUCER_THREADS, CONSUMER_THREADS);

    // 正确性验证
    cudaMemset(d_out, 0, N * sizeof(float));
    pipe_vec_warp_specialized<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_in, d_out, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_out.data(), h_ref.data(), N, "Warp特化");

    // 性能测试
    for (int i = 0; i < warmup_iters; i++) {
        pipe_vec_warp_specialized<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++) {
        pipe_vec_warp_specialized<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float ws_avg_ms = elapsed_ms / bench_iters;
    double ws_bw = data_bytes / (ws_avg_ms * 1e-3) / 1e9;
    printf("[Warp特化] Avg Time: %.4f ms, Bandwidth: %.1f GB/s\n", ws_avg_ms, ws_bw);

    // ====================================================================
    // 性能汇总
    // ====================================================================
    printf("\n========== 性能汇总 (N=%d, %d iterations) ==========\n", N, bench_iters);
    printf("  %-20s  %10s  %12s\n", "Kernel", "Avg (ms)", "BW (GB/s)");
    printf("  %-20s  %10.4f  %12.1f\n", "朴素版本",     naive_avg_ms, naive_bw);
    printf("  %-20s  %10.4f  %12.1f\n", "统一线程流水线", pipe_avg_ms,  pipe_bw);
    printf("  %-20s  %10.4f  %12.1f\n", "Warp特化流水线", ws_avg_ms,    ws_bw);
    printf("  --------------------------------------------------\n");
    printf("  统一线程 vs 朴素: %.2fx\n", naive_avg_ms / pipe_avg_ms);
    printf("  Warp特化 vs 朴素: %.2fx\n", naive_avg_ms / ws_avg_ms);
    printf("  Warp特化 vs 统一线程: %.2fx\n", pipe_avg_ms / ws_avg_ms);

    // --- 释放资源 ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}