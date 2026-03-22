/**
 * pipe_tma.cu
 *
 * 演示 Hopper (SM90) 上基于 TMA 的流水线优化，包含三种实现的性能对比：
 *
 *   1. 朴素版本 (naive)：直接从 GMEM 读取，计算后写回，无任何优化
 *   2. 统一线程 TMA 流水线：所有线程交替充当 Producer/Consumer，
 *      通过 TMA 异步加载数据到 SMEM，再从 SMEM 计算写回
 *   3. Warp 特化 TMA 流水线：Warp 0 专职通过 TMA 搬运数据，
 *      Warp 1-3 专职计算，双方通过 mbarrier 同步形成 pipeline
 *
 * 对比维度：
 *   - 朴素 vs TMA 流水线：TMA 异步拷贝带来的访存延迟隐藏
 *   - 统一线程 vs Warp 特化：Producer/Consumer 并行 vs 串行的收益
 *
 * 依赖：CUTLASS 3.x / CuTe, Hopper (SM90) GPU
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

// ============================================================================
// 超参数配置
// ============================================================================
static constexpr int COMPUTE_ITERS = 20;                             // 每个元素的计算迭代次数（用于模拟计算负载）
static constexpr int PIPE_STAGES   = 2;                             // 流水线 stage 数
static constexpr int BLOCK_SIZE    = 128;                           // 每个 CTA 的线程数
static constexpr int TILE_SIZE     = BLOCK_SIZE;                    // 每个 tile 处理的元素数
static constexpr int SMEM_PER_STAGE = TILE_SIZE * sizeof(float);    // 每个 stage 的共享内存大小（字节）

static constexpr int PRODUCER_THREADS = 32;   // Producer: Warp 0（负责 TMA 搬运）
static constexpr int CONSUMER_THREADS = 96;   // Consumer: Warp 1-3（负责计算）

// ============================================================================
// 模拟计算负载（device / host 版本保持一致，用于结果验证）
// ============================================================================
__device__ __forceinline__ float compute_func(float val) {
    float result = val;
    #pragma unroll 1
    for (int iter = 0; iter < COMPUTE_ITERS; ++iter) {
        result = sinf(result) * expf(-result * 0.01f) + val;
    }
    return result;
}

inline float compute_func_host(float val) {
    float result = val;
    for (int iter = 0; iter < COMPUTE_ITERS; ++iter) {
        result = sinf(result) * expf(-result * 0.01f) + val;
    }
    return result;
}

// ============================================================================
// Shared Memory 布局（统一线程流水线和 Warp 特化流水线共用）
// ============================================================================
struct SharedStorage {
    alignas(128) float buf[PIPE_STAGES][TILE_SIZE];   // 数据缓冲区，每个 stage 一个 tile
    uint64_t producer_mbar[PIPE_STAGES];  // Producer barrier：通知 Consumer 数据已到达 SMEM
    uint64_t consumer_mbar[PIPE_STAGES];  // Consumer barrier：通知 Producer 该 stage 已消费完毕可复用
};

// ============================================================================
// Kernel 0：朴素版本（无流水线，无 Shared Memory）
//
// 最简单的 baseline：每个线程直接从 GMEM 读取，计算后写回。
// 无 SMEM、无 TMA、无 pipeline，用于性能对比的下界参考。
// ============================================================================
__global__ void naive_kernel(const float* __restrict__ in,
                             float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        out[i] = compute_func(in[i]);
    }
}

// ============================================================================
// Kernel 1：统一线程 TMA 流水线
//
// 所有 128 个线程交替充当 Producer 和 Consumer（串行执行）：
//   1. Prefetch 阶段：通过 TMA 预填充所有 pipeline stage
//   2. 主循环：Consumer 消费当前 stage → Producer 加载下一个 tile
//
// 与 Warp 特化版本的关键区别：
//   - 所有线程同时参与 load 和 compute，无法重叠
//   - 需要 __syncthreads 协调全部线程，开销更大
//   - TMA 仅由 lane 0 发起，其余线程等待 mbarrier
//
// 流水线时序：
//   All threads: [TMA Load S0] [TMA Load S1] [Compute S0, TMA Load S0'] [Compute S1, ...] ...
//   注意：load 和 compute 是串行的，pipeline 的收益在于 TMA 的异步性
// ============================================================================
template <class TmaLoad_>
__global__ void
pipe_tma_unified_kernel(float* out, int N, int total_tiles,
                        CUTLASS_GRID_CONSTANT TmaLoad_ const tma_load)
{
    extern __shared__ char shared_memory[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    int tid      = threadIdx.x;
    int lane_id  = tid % 32;

    // 统一线程模式：所有线程参与 barrier，TMA 由 lane 0 发起
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    uint64_t* producer_mbar = smem.producer_mbar;
    uint64_t* consumer_mbar = smem.consumer_mbar;
    constexpr uint32_t tma_transaction_bytes = TILE_SIZE * sizeof(float);

    // 初始化 mbarrier
    //   producer_mbar: 到达者 = 1（TMA 硬件单元自动 arrive）
    //   consumer_mbar: 到达者 = BLOCK_SIZE（所有线程参与消费后 arrive）
    if (tid == 0) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], BLOCK_SIZE);
        }
    }
    __syncthreads();

    // 计算当前 CTA 负责的 tile 数（grid-stride 分配）
    int my_tile_count = 0;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        my_tile_count++;
    }

    // 构建 TMA 源张量
    Tensor mIn = tma_load.get_tma_tensor(make_shape(total_tiles * Int<TILE_SIZE>{}));
    auto cta_tma    = tma_load.get_slice(Int<0>{});
    Tensor tAgA_all = cta_tma.partition_S(mIn);

    auto write_state = cutlass::PipelineState<PIPE_STAGES>();
    auto read_state  = cutlass::PipelineState<PIPE_STAGES>();
    int k_tile_count = my_tile_count;
    int load_tile    = blockIdx.x;
    int store_tile   = blockIdx.x;

    // Prefetch 阶段：通过 TMA 异步预填充所有 pipeline stage
    CUTE_UNROLL
    for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
        if (k_tile_count > 0) {
            if (tid == 0) {
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);

                Tensor sA_pipe = make_tensor(make_smem_ptr(smem.buf[pipe]), make_shape(Int<TILE_SIZE>{}));
                Tensor tAsA_pipe = cta_tma.partition_D(sA_pipe);

                copy(tma_load.with(producer_mbar[pipe]),
                     tAgA_all(_, load_tile), tAsA_pipe(_, 0));
            }
            --k_tile_count;
            load_tile += gridDim.x;
        }
    }

    // 重置计数器用于消费
    k_tile_count = my_tile_count;

    // 主循环：串行地 消费当前 stage → 加载下一个 tile
    CUTE_NO_UNROLL
    while (k_tile_count > -PIPE_STAGES) {
        // ---- Consumer 阶段：等待 TMA 完成，从 SMEM 计算并写回 GMEM ----
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        int global_offset = store_tile * TILE_SIZE;
        if (global_offset + tid < N) {
            float val = compute_func(smem.buf[read_pipe][tid]);
            out[global_offset + tid] = val;
        }
        store_tile += gridDim.x;

        __syncthreads();

        // 所有线程 arrive consumer_mbar，释放当前 stage
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        // ---- Producer 阶段：等待 Consumer 释放 stage → TMA 加载新 tile ----
        if (k_tile_count > 0) {
            int write_pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[write_pipe], write_state.phase());

            if (tid == 0) {
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[write_pipe], tma_transaction_bytes);

                Tensor sA_pipe = make_tensor(make_smem_ptr(smem.buf[write_pipe]), make_shape(Int<TILE_SIZE>{}));
                Tensor tAsA_pipe = cta_tma.partition_D(sA_pipe);

                copy(tma_load.with(producer_mbar[write_pipe]),
                     tAgA_all(_, load_tile), tAsA_pipe(_, 0));
            }

            ++write_state;
            load_tile += gridDim.x;
        }

        --k_tile_count;
    }
}

// ============================================================================
// Kernel 2：Warp 特化 TMA 流水线
//
// Warp 0 (Producer) 和 Warp 1-3 (Consumer) 走完全不同的代码路径，
// 通过 mbarrier 实现真正的 Producer/Consumer 并行：
//   - Producer 通过 TMA 异步搬运数据，不占用计算资源
//   - Consumer 在等待数据时可以处理上一个 stage 的尾部工作
//   - 初始化后不再使用 __syncthreads，完全依赖 mbarrier 实现无锁流水线
//
// 流水线时序示意（以 2 stage 为例）：
//
//   Producer:  [Load tile0 → S0] [Load tile1 → S1] [Wait C0, Load tile2 → S0] ...
//   Consumer:                     [Wait P0, Compute S0] [Wait P1, Compute S1] ...
//
//   P = producer_mbar (TMA 完成后自动 arrive)
//   C = consumer_mbar (Consumer 计算完后 arrive，通知 Producer 可复用)
// ============================================================================
template <class TmaLoad_>
__global__ void
pipe_tma_warp_specialized_kernel(float* out, int N, int total_tiles,
                                 CUTLASS_GRID_CONSTANT TmaLoad_ const tma_load)
{
    extern __shared__ char shared_memory[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    int tid       = threadIdx.x;
    int warp_idx  = tid / 32;
    int lane_id   = tid % 32;
    bool is_producer = (warp_idx == 0);

    // ClusterTransactionBarrier：支持 expect_tx，配合 TMA 自动 arrive
    // ClusterBarrier：普通 arrive/wait barrier
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    uint64_t* producer_mbar = smem.producer_mbar;
    uint64_t* consumer_mbar = smem.consumer_mbar;
    constexpr uint32_t tma_transaction_bytes = TILE_SIZE * sizeof(float);

    // 初始化 mbarrier
    //   producer_mbar: 到达者计数 = 1（仅 TMA 硬件单元 arrive）
    //   consumer_mbar: 到达者计数 = CONSUMER_THREADS（所有 Consumer 线程 arrive）
    if (warp_idx == 0 && lane_id == 0) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], CONSUMER_THREADS);
        }
    }
    __syncthreads();  // 初始化后的唯一一次全 block 同步，此后完全依赖 mbarrier

    // 计算当前 CTA 负责的 tile 数（grid-stride 分配）
    int my_tile_count = 0;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        my_tile_count++;
    }

    // 构建 TMA 源张量：全局内存的 1D 视图，按 TILE_SIZE 分 tile
    Tensor mIn = tma_load.get_tma_tensor(make_shape(total_tiles * Int<TILE_SIZE>{}));
    auto cta_tma    = tma_load.get_slice(Int<0>{});
    Tensor tAgA_all = cta_tma.partition_S(mIn);

    // ====================================================================
    // Warp 特化分叉
    // ====================================================================
    if (is_producer) {
        // ================================================================
        // Producer (Warp 0)：通过 TMA 将数据从 GMEM 加载到 SMEM
        // ================================================================
        auto write_state = cutlass::PipelineState<PIPE_STAGES>();
        int k_tile_count = my_tile_count;
        int load_tile    = blockIdx.x;

        // Prefetch 阶段：填满所有流水线 stage
        CUTE_UNROLL
        for (int pipe = 0; pipe < PIPE_STAGES; ++pipe) {
            if (k_tile_count > 0) {
                if (lane_id == 0) {
                    // 声明即将有 tma_transaction_bytes 字节通过 TMA 到达
                    ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);

                    // 构建当前 stage 的 SMEM tile 视图并发起 TMA 拷贝
                    Tensor sA_pipe = make_tensor(make_smem_ptr(smem.buf[pipe]), make_shape(Int<TILE_SIZE>{}));
                    Tensor tAsA_pipe = cta_tma.partition_D(sA_pipe);

                    copy(tma_load.with(producer_mbar[pipe]),
                         tAgA_all(_, load_tile), tAsA_pipe(_, 0));
                }
                --k_tile_count;
                load_tile += gridDim.x;
            }
        }

        // 主循环：等待 Consumer 释放 stage → 加载下一个 tile
        CUTE_NO_UNROLL
        while (k_tile_count > 0) {
            int write_pipe = write_state.index();

            if (lane_id == 0) {
                // 等待 Consumer 消费完当前 stage
                ConsumerBarType::wait(&consumer_mbar[write_pipe], write_state.phase());
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[write_pipe], tma_transaction_bytes);

                Tensor sA_pipe = make_tensor(make_smem_ptr(smem.buf[write_pipe]), make_shape(Int<TILE_SIZE>{}));
                Tensor tAsA_pipe = cta_tma.partition_D(sA_pipe);

                copy(tma_load.with(producer_mbar[write_pipe]), tAgA_all(_, load_tile), tAsA_pipe(_, 0));
            }

            ++write_state;
            --k_tile_count;
            load_tile += gridDim.x;
        }

    } else {
        // ================================================================
        // Consumer (Warp 1-3)：从 SMEM 读取数据，计算后写回 GMEM
        // ================================================================
        int consumer_tid = tid - PRODUCER_THREADS;  // Consumer 内部线程编号 [0, 96)
        auto read_state  = cutlass::PipelineState<PIPE_STAGES>();
        int k_tile_count = my_tile_count;
        int store_tile   = blockIdx.x;

        CUTE_NO_UNROLL
        while (k_tile_count > 0) {
            int read_pipe = read_state.index();

            // 等待 Producer 完成当前 stage 的 TMA 加载
            ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

            // 从 SMEM 读取数据，执行计算，写回 GMEM
            int global_offset = store_tile * TILE_SIZE;
            for (int i = consumer_tid; i < TILE_SIZE; i += CONSUMER_THREADS) {
                float val = compute_func(smem.buf[read_pipe][i]);
                if (global_offset + i < N) {
                    out[global_offset + i] = val;
                }
            }

            // 通知 Producer 当前 stage 已消费完毕，可以复用
            ConsumerBarType::arrive(&consumer_mbar[read_pipe]);

            ++read_state;
            --k_tile_count;
            store_tile += gridDim.x;
        }
    }
}

// ============================================================================
// 验证函数
// ============================================================================
void verify_result(const float* h_out, const float* h_ref, int N, const char* label) {
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(h_out[i] - h_ref[i]);
        if (diff > 1e-5f) {
            if (error_count < 5) printf("  [%s] Mismatch at [%d]: got %.6f, exp %.6f\n", label, i, h_out[i], h_ref[i]);
            error_count++;
        }
        max_error = std::fmax(max_error, diff);
    }
    if (error_count == 0) printf("[%s] Result: PASSED (max error: %.6f)\n", label, max_error);
    else printf("[%s] Result: FAILED (%d errors, max error: %.6f)\n", label, error_count, max_error);
}

// ============================================================================
// Main：正确性验证 + 性能对比（朴素 vs 统一线程 TMA 流水线 vs Warp 特化 TMA 流水线）
// ============================================================================
int main(int argc, char** argv) {
    int N = 1024 * 1024 * 128;
    if (argc >= 2) N = atoi(argv[1]);

    int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int N_padded = total_tiles * TILE_SIZE;  // 对齐到 TILE_SIZE，避免 TMA 越界访问

    printf("TMA 流水线性能对比：朴素 vs 统一线程 TMA vs Warp 特化 TMA\n");
    printf("N = %d (%.2f MB), Tiles = %d, Stages = %d\n\n",
           N, N * sizeof(float) / (1024.0f * 1024.0f), total_tiles, PIPE_STAGES);

    // 准备 host 端数据和参考结果
    std::vector<float> h_in(N_padded, 0.0f), h_ref(N, 0.0f), h_out(N, 0.0f);
    srand(42);
    // for (int i = 0; i < N; i++) {
    //     h_in[i] = static_cast<float>(rand() % 100) / 10.0f;
    //     h_ref[i] = compute_func_host(h_in[i]);
    // }

    // 分配 device 内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, N_padded * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_in, 0, N_padded * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N_padded * sizeof(float), cudaMemcpyHostToDevice);

    // 构建 CuTe 全局内存张量（1D 连续布局）
    auto gmem_layout = make_layout(make_shape(N_padded));
    Tensor gIn = make_tensor(d_in, gmem_layout);

    // 构建 SMEM tile 布局，TMA 按此粒度搬运
    auto smem_layout = make_layout(make_shape(Int<TILE_SIZE>{}));

    // 创建 TMA descriptor：描述 GMEM → SMEM 的异步拷贝模式
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gIn, smem_layout);
    int num_blocks = std::min(total_tiles, 256);
    int smem_size  = sizeof(SharedStorage);

    // CUDA Event 用于精确计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int warmup_iters = 10;
    const int bench_iters  = 100;
    float elapsed_ms = 0.0f;

    // 数据量（读 + 写）用于计算等效带宽
    double data_bytes = 2.0 * N * sizeof(float);

    float naive_avg_ms = 0.0f, unified_avg_ms = 0.0f, ws_avg_ms = 0.0f;
    double naive_bw = 0.0, unified_bw = 0.0, ws_bw = 0.0;

    // ====================================================================
    // 测试 0：朴素版本 (naive_kernel)
    // ====================================================================
    printf("=== 测试 0：朴素版本 (naive_kernel) ===\n");
    printf("Blocks: %d, Threads: %d, 无 Shared Memory, 无 TMA\n", num_blocks, BLOCK_SIZE);

    cudaMemset(d_out, 0, N * sizeof(float));
    naive_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_out.data(), h_ref.data(), N, "朴素版本");

    for (int i = 0; i < warmup_iters; i++)
        naive_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++)
        naive_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    naive_avg_ms = elapsed_ms / bench_iters;
    naive_bw = data_bytes / (naive_avg_ms * 1e-3) / 1e9;
    printf("[朴素版本] Avg: %.4f ms, BW: %.1f GB/s\n", naive_avg_ms, naive_bw);

    // ====================================================================
    // 测试 1：统一线程 TMA 流水线 (pipe_tma_unified_kernel)
    // ====================================================================
    printf("\n=== 测试 1：统一线程 TMA 流水线 (pipe_tma_unified_kernel) ===\n");
    printf("Blocks: %d, Threads: %d, Smem: %d bytes\n", num_blocks, BLOCK_SIZE, smem_size);
    printf("所有线程交替 Producer/Consumer，TMA 由 tid 0 发起\n");

    cudaMemset(d_out, 0, N * sizeof(float));
    pipe_tma_unified_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_out, N, total_tiles, tma_load);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_out.data(), h_ref.data(), N, "统一线程TMA");

    for (int i = 0; i < warmup_iters; i++)
        pipe_tma_unified_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_out, N, total_tiles, tma_load);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++)
        pipe_tma_unified_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_out, N, total_tiles, tma_load);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    unified_avg_ms = elapsed_ms / bench_iters;
    unified_bw = data_bytes / (unified_avg_ms * 1e-3) / 1e9;
    printf("[统一线程TMA] Avg: %.4f ms, BW: %.1f GB/s\n", unified_avg_ms, unified_bw);

    // ====================================================================
    // 测试 2：Warp 特化 TMA 流水线 (pipe_tma_warp_specialized_kernel)
    // ====================================================================
    printf("\n=== 测试 2：Warp 特化 TMA 流水线 (pipe_tma_warp_specialized_kernel) ===\n");
    printf("Blocks: %d, Threads: %d, Smem: %d bytes\n", num_blocks, BLOCK_SIZE, smem_size);
    printf("Producer: Warp 0 (%d threads, TMA), Consumer: Warp 1-3 (%d threads)\n",
           PRODUCER_THREADS, CONSUMER_THREADS);

    cudaMemset(d_out, 0, N * sizeof(float));
    pipe_tma_warp_specialized_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_out, N, total_tiles, tma_load);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_out.data(), h_ref.data(), N, "Warp特化TMA");

    for (int i = 0; i < warmup_iters; i++)
        pipe_tma_warp_specialized_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_out, N, total_tiles, tma_load);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++)
        pipe_tma_warp_specialized_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(d_out, N, total_tiles, tma_load);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    ws_avg_ms = elapsed_ms / bench_iters;
    ws_bw = data_bytes / (ws_avg_ms * 1e-3) / 1e9;
    printf("[Warp特化TMA] Avg: %.4f ms, BW: %.1f GB/s\n", ws_avg_ms, ws_bw);

    // ====================================================================
    // 性能汇总
    // ====================================================================
    printf("\n========== 性能汇总 (N=%d, %d iterations) ==========\n", N, bench_iters);
    printf("  %-24s  %10s  %12s\n", "Kernel", "Avg (ms)", "BW (GB/s)");
    printf("  %-24s  %10.4f  %12.1f\n", "朴素版本",           naive_avg_ms,   naive_bw);
    printf("  %-24s  %10.4f  %12.1f\n", "统一线程 TMA 流水线", unified_avg_ms, unified_bw);
    printf("  %-24s  %10.4f  %12.1f\n", "Warp特化 TMA 流水线", ws_avg_ms,      ws_bw);
    printf("  --------------------------------------------------------\n");
    printf("  统一线程TMA vs 朴素:     %.2fx\n", naive_avg_ms / unified_avg_ms);
    printf("  Warp特化TMA vs 朴素:     %.2fx\n", naive_avg_ms / ws_avg_ms);
    printf("  Warp特化TMA vs 统一线程: %.2fx\n", unified_avg_ms / ws_avg_ms);

    // 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
