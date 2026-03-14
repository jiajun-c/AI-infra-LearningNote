#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <iostream>

using namespace cute;

// ==========================================
// 1. 发送方依然需要这个核心推送函数
// ==========================================
// 因为 CuTe 目前主要封装了 Copy (搬运) 和 Wait (等待)，
// 对于这种"Scalar Store + Remote Barrier Update"的推送操作，
// 手写这个 PTX 依然是最直接最高效的方法。
__device__ inline void store_shared_remote_f32(float value, uint32_t dsmem_addr, uint32_t remote_barrier_addr) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                 : : "r"(dsmem_addr), "f"(value), "r"(remote_barrier_addr));
}

// ==========================================
// 2. 地址计算辅助 (保持不变)
// ==========================================
__device__ inline uint32_t get_remote_offset(void* local_ptr, int target_rank) {
    uint32_t smem_offset = __cvta_generic_to_shared(local_ptr);
    return set_block_rank(smem_offset, target_rank);
}

// ==========================================
// Kernel: 使用 ClusterTransactionBarrier
// ==========================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__global__ void __cluster_dims__(2, 1, 1) cluster_push_simplified(float* debug_out)
#else
__global__ void cluster_push_simplified(float* debug_out)
#endif
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 900
    if (threadIdx.x == 0) printf("Error: Requires SM90a architecture\n");
    return;
#endif

    uint32_t rank = block_rank_in_cluster();
    
    // Shared Memory 布局
    // 使用 CuTe 提供的 Barrier 类型，无需自己算偏移
    // 这里的 ValueType=float 表示我们主要用它来同步 float 数据的传输
    extern __shared__ char smem_raw[];
    float* local_data = (float*) smem_raw;
    // Barrier 紧跟在数据后面 (注意对齐)
    using BarrierType = float; // 仅用于标记类型，实际Barrier占8字节
    // 使用 CuTe 封装类: ClusterTransactionBarrier
    cutlass::arch::ClusterTransactionBarrier* barrier = reinterpret_cast< cutlass::arch::ClusterTransactionBarrier*>(smem_raw + 16);

    // ========================================================
    // 接收方 (Rank 1)
    // ========================================================
    if (rank == 1 && threadIdx.x == 0) {
        // 1. 初始化 (替代了 mbarrier_init)
        // 参数 1: 参与等待的线程数 (这里只有主线程在等)
        barrier->init(1); 
        
        // 2. 期待事务 (替代了 mbarrier_expect_bytes)
        // 告诉 Barrier 我期待收到多少字节
        barrier->arrive_and_expect_tx(sizeof(float));
        
        *local_data = 0.0f; // 清零以便验证
    }

    // Cluster 同步
    cluster_sync();

    // ========================================================
    // 发送方 (Rank 0)
    // ========================================================
    if (rank == 0 && threadIdx.x == 0) {
        float val_to_send = 99.99f;

        // 计算 Rank 1 的地址
        uint32_t target_data = get_remote_offset(local_data, 1);
        
        // 获取 Barrier 的原始指针 (generic ptr) 并转换为 remote offset
        // barrier->ptr() 返回 generic 指针
        uint32_t target_barrier = get_remote_offset(reinterpret_cast<void*>(barrier), 1);

        // 推送！
        store_shared_remote_f32(val_to_send, target_data, target_barrier);
        
        printf("Rank 0: Pushed %f\n", val_to_send);
    }

    // ========================================================
    // 接收方 (Rank 1) - 等待
    // ========================================================
    if (rank == 1 && threadIdx.x == 0) {
        // 3. 等待 (替代了 mbarrier_wait)
        // 核心简化点：你贴出的代码正是这个 wait 函数的底层实现
        // 它自动处理 Phase 0/1 翻转和循环等待
        
        // 注意：mbarrier 需要一个 Phase 参数 (0或1)
        // 对于第一次使用，Phase 通常是 0
        int phase = 0; 
        barrier->wait(phase);

        float val = *local_data;
        printf("Rank 1: Barrier released! Received: %f\n", val);
        debug_out[0] = val;
    }
}

// ==========================================
// Host 代码
// ==========================================
int main() {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));

    // 2 个 Block 组成 Cluster
    dim3 blocks(2);
    dim3 threads(32);
    // SMEM 大小：4字节数据 + 4字节padding + 8字节barrier = 16字节 (给1024更安全)
    int smem_size = 1024;

    printf("Launching Solution Kernel...\n");
    cluster_push_simplified<<<blocks, threads, smem_size>>>(d_out);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    float h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    if (abs(h_out - 123.456f) < 1e-5) {
        printf("SUCCESS: Problem Solved!\n");
    } else {
        printf("FAILED: Got %f\n", h_out);
    }

    cudaFree(d_out);
    return 0;
}