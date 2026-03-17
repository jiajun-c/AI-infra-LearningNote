#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 定义流水线级数
constexpr int STAGES = 3;

// -------------------------------------------------------------------------
// Baseline Kernel: 无流水线，使用传统 Global → Register → Shared 方式
// 每次迭代同步加载一块数据到共享内存，再计算，再写回
// -------------------------------------------------------------------------
__global__ void naive_no_pipeline_kernel(const float4* __restrict__ g_in,
                                          float4* __restrict__ g_out,
                                          int tiles_per_block)
{
    // 共享内存：只需 1 个 Buffer（无流水线，不需要多 stage）
    extern __shared__ float4 s_data[];

    int tid = threadIdx.x;
    int tile_size = blockDim.x;
    int block_offset = blockIdx.x * tiles_per_block * tile_size;

    for (int i = 0; i < tiles_per_block; ++i) {
        // Step 1: 同步加载 Global → Register → Shared（传统路径）
        //         数据经过寄存器中转，无法和计算重叠
        s_data[tid] = g_in[block_offset + i * tile_size + tid];
        __syncthreads();

        // Step 2: 从 Shared Memory 读取并计算
        float4 val = s_data[tid];
        val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;

        // Step 3: 写回 Global Memory
        g_out[block_offset + i * tile_size + tid] = val;
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// Baseline Kernel 2: 无流水线，使用 cp.async 但每次加载后立即等待（无重叠）
// 展示 cp.async 本身的好处（绕过寄存器），但没有流水线带来的计算/访存重叠
// -------------------------------------------------------------------------
__global__ void no_pipeline_cpasync_kernel(const float4* __restrict__ g_in,
                                            float4* __restrict__ g_out,
                                            int tiles_per_block)
{
    extern __shared__ float4 s_data[];

    int tid = threadIdx.x;
    int tile_size = blockDim.x;
    int block_offset = blockIdx.x * tiles_per_block * tile_size;

    for (int i = 0; i < tiles_per_block; ++i) {
        // Step 1: 使用 cp.async 异步拷贝 Global → Shared（绕过寄存器）
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&s_data[tid]));
        const float4* global_ptr = &g_in[block_offset + i * tile_size + tid];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(global_ptr));
        asm volatile("cp.async.commit_group;\n" ::);

        // Step 2: 立即等待拷贝完成 —— 没有流水线，完全阻塞
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        // Step 3: 计算
        float4 val = s_data[tid];
        val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;

        // Step 4: 写回
        g_out[block_offset + i * tile_size + tid] = val;
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// CUDA Kernel: 3 级流水线 + float4 向量化异步拷贝
// -------------------------------------------------------------------------
__global__ void pipeline_cp_async_kernel(const float4* __restrict__ g_in, 
                                         float4* __restrict__ g_out, 
                                         int tiles_per_block) 
{
    // 共享内存：划分为 STAGES 个 Buffer
    // 假设每个 Block 有 128 个线程，每个线程处理 1 个 float4 (即 4 个 float)
    // TILE_SIZE = 128 个 float4
    extern __shared__ float4 s_data[]; 

    int tid = threadIdx.x;
    int tile_size = blockDim.x; 

    // 该 Block 在全局内存中的起始物理偏移
    int block_offset = blockIdx.x * tiles_per_block * tile_size;

    // ==========================================
    // 阶段 1：Prologue (序幕) - 填满流水线的初始阶段
    // ==========================================
    // 发起第 0 块数据的拷贝 (放入 Buffer 0)
    uint32_t smem_ptr_0 = static_cast<uint32_t>(__cvta_generic_to_shared(&s_data[0 * tile_size + tid]));
    const float4* global_ptr_0 = &g_in[block_offset + 0 * tile_size + tid];
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr_0), "l"(global_ptr_0));
    asm volatile("cp.async.commit_group;\n" ::); // 提交第 1 组，当前在途组数：1

    // 发起第 1 块数据的拷贝 (放入 Buffer 1)
    uint32_t smem_ptr_1 = static_cast<uint32_t>(__cvta_generic_to_shared(&s_data[1 * tile_size + tid]));
    const float4* global_ptr_1 = &g_in[block_offset + 1 * tile_size + tid];
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr_1), "l"(global_ptr_1));
    asm volatile("cp.async.commit_group;\n" ::); // 提交第 2 组，当前在途组数：2

    // ==========================================
    // 阶段 2：Mainloop (主循环) - 算发并举
    // ==========================================
    // 循环处理，直到剩下最后 2 块数据
    for (int i = 0; i < tiles_per_block - 2; ++i) {
        // 1. 发起第 i+2 块数据的拷贝 (循环放入 Buffer)
        int load_stage_idx = (i + 2) % STAGES;
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&s_data[load_stage_idx * tile_size + tid]));
        const float4* global_ptr = &g_in[block_offset + (i + 2) * tile_size + tid];
        
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(global_ptr));
        asm volatile("cp.async.commit_group;\n" ::); // 提交后，当前在途组数：3

        // 2. 阻塞等待：直到在途组数 <= 2 (即最老的那组数据必须到了)
        asm volatile("cp.async.wait_group 2;\n" ::);
        __syncthreads(); // 必须同步，确保大家都看到了到达的数据

        // 3. 计算第 i 块数据
        int compute_stage_idx = i % STAGES;
        float4 val = s_data[compute_stage_idx * tile_size + tid];
        
        // 模拟计算负载：乘 2
        val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;
        g_out[block_offset + i * tile_size + tid] = val;

        // 4. 同步：防止算得快的线程进入下一轮把共享内存覆盖掉
        __syncthreads();
    }

    // ==========================================
    // 阶段 3：Epilogue (尾声) - 清空流水线
    // ==========================================
    // 处理倒数第 2 块数据
    asm volatile("cp.async.wait_group 1;\n" ::); // 等待直到在途组数 <= 1
    __syncthreads();
    int idx_n2 = tiles_per_block - 2;
    float4 val_n2 = s_data[idx_n2 % STAGES * tile_size + tid];
    val_n2.x *= 2.0f; val_n2.y *= 2.0f; val_n2.z *= 2.0f; val_n2.w *= 2.0f;
    g_out[block_offset + idx_n2 * tile_size + tid] = val_n2;
    __syncthreads();

    // 处理最后 1 块数据
    asm volatile("cp.async.wait_group 0;\n" ::); // 等待所有组都到达
    __syncthreads();
    int idx_n1 = tiles_per_block - 1;
    float4 val_n1 = s_data[idx_n1 % STAGES * tile_size + tid];
    val_n1.x *= 2.0f; val_n1.y *= 2.0f; val_n1.z *= 2.0f; val_n1.w *= 2.0f;
    g_out[block_offset + idx_n1 * tile_size + tid] = val_n1;
}

// -------------------------------------------------------------------------
// 辅助函数：验证输出结果
// -------------------------------------------------------------------------
bool verify(const std::vector<float>& h_in, const std::vector<float>& h_out, int num_floats, const char* name) {
    for (int i = 0; i < num_floats; ++i) {
        if (h_out[i] != h_in[i] * 2.0f) {
            std::cerr << "[" << name << "] Mismatch at index " << i
                      << " expected=" << h_in[i] * 2.0f << " got=" << h_out[i] << std::endl;
            return false;
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// 辅助函数：用 CUDA Event 测量 kernel 的平均耗时
// -------------------------------------------------------------------------
template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel_launch, int warmup_iters = 10, int bench_iters = 100) {
    // Warmup：让 GPU 进入稳定工作状态
    for (int i = 0; i < warmup_iters; ++i) {
        kernel_launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        kernel_launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / bench_iters;
}

// -------------------------------------------------------------------------
// Host 端主函数
// -------------------------------------------------------------------------
int main() {
    // ============================================================
    // 参数配置 —— 使用更大的数据量以凸显性能差异
    // ============================================================
    const int threads_per_block = 128;
    const int tiles_per_block = 64;   // 每个 Block 处理 64 块数据（必须 >= 3）
    const int num_blocks = 256;

    // 总元素个数 (1 个 float4 包含 4 个 float)
    const int num_float4 = num_blocks * tiles_per_block * threads_per_block;
    const int num_floats = num_float4 * 4;
    const int bytes = num_floats * sizeof(float);

    printf("============================================================\n");
    printf("  cp.async Pipeline vs No-Pipeline 性能对比\n");
    printf("============================================================\n");
    printf("配置: %d blocks × %d tiles/block × %d threads/block\n",
           num_blocks, tiles_per_block, threads_per_block);
    printf("数据量: %d floats = %.2f MB\n", num_floats, bytes / (1024.0 * 1024.0));
    printf("------------------------------------------------------------\n\n");

    // 分配并初始化 Host 内存
    std::vector<float> h_in(num_floats);
    std::vector<float> h_out(num_floats, 0.0f);
    for (int i = 0; i < num_floats; ++i) h_in[i] = static_cast<float>(i % 1000);

    // 分配 Device 内存 (按 float4 指针操作)
    float4 *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // 共享内存大小
    size_t smem_pipeline = STAGES * threads_per_block * sizeof(float4);   // 3-stage 流水线
    size_t smem_single   = 1     * threads_per_block * sizeof(float4);    // 单 buffer（无流水线）

    // ============================================================
    // Kernel 1: 朴素版 —— 传统 Global→Register→Shared，无流水线
    // ============================================================
    {
        printf("[Kernel 1] Naive (Global→Reg→Shared, 无流水线)\n");
        // 正确性验证
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        naive_no_pipeline_kernel<<<num_blocks, threads_per_block, smem_single>>>(d_in, d_out, tiles_per_block);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_in, h_out, num_floats, "Naive");
        printf("  正确性: %s\n", ok ? "PASS ✓" : "FAIL ✗");

        // 性能测量
        float avg_ms = benchmark_kernel([&]() {
            naive_no_pipeline_kernel<<<num_blocks, threads_per_block, smem_single>>>(d_in, d_out, tiles_per_block);
        });
        printf("  平均耗时: %.4f ms\n\n", avg_ms);
    }

    // ============================================================
    // Kernel 2: cp.async 无流水线 —— 绕过寄存器，但加载后立即等待
    // ============================================================
    {
        printf("[Kernel 2] cp.async 无流水线 (每次加载后立即 wait)\n");
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        no_pipeline_cpasync_kernel<<<num_blocks, threads_per_block, smem_single>>>(d_in, d_out, tiles_per_block);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_in, h_out, num_floats, "cp.async no-pipe");
        printf("  正确性: %s\n", ok ? "PASS ✓" : "FAIL ✗");

        float avg_ms = benchmark_kernel([&]() {
            no_pipeline_cpasync_kernel<<<num_blocks, threads_per_block, smem_single>>>(d_in, d_out, tiles_per_block);
        });
        printf("  平均耗时: %.4f ms\n\n", avg_ms);
    }

    // ============================================================
    // Kernel 3: cp.async + 3-Stage 流水线 —— 计算与访存重叠
    // ============================================================
    {
        printf("[Kernel 3] cp.async + 3-Stage Pipeline (计算与访存重叠)\n");
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        pipeline_cp_async_kernel<<<num_blocks, threads_per_block, smem_pipeline>>>(d_in, d_out, tiles_per_block);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify(h_in, h_out, num_floats, "Pipeline");
        printf("  正确性: %s\n", ok ? "PASS ✓" : "FAIL ✗");

        float avg_ms = benchmark_kernel([&]() {
            pipeline_cp_async_kernel<<<num_blocks, threads_per_block, smem_pipeline>>>(d_in, d_out, tiles_per_block);
        });
        printf("  平均耗时: %.4f ms\n\n", avg_ms);
    }

    // ============================================================
    // 汇总
    // ============================================================
    printf("============================================================\n");
    printf("  说明\n");
    printf("============================================================\n");
    printf("Kernel 1 (Naive):          传统路径 Global→Reg→Shared，数据\n");
    printf("                           经寄存器中转，加载与计算完全串行。\n\n");
    printf("Kernel 2 (cp.async 无pipe): 使用 cp.async 绕过寄存器直达\n");
    printf("                           Shared Memory，但每次加载后立即\n");
    printf("                           wait，无法重叠计算与访存。\n\n");
    printf("Kernel 3 (3-Stage Pipe):   cp.async + 3级流水线，在等待第i\n");
    printf("                           块数据时已发起第i+2块的预取，\n");
    printf("                           实现计算与访存的深度重叠。\n");
    printf("============================================================\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}