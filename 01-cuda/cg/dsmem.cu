#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cooperative_groups;

// ==========================================
// 1. 辅助函数实现
// ==========================================

// 初始化本地 Shared Memory
__device__ void init_shared_data(const thread_block& block, int *data) {
    if (block.thread_rank() == 0) {
        // 获取当前 Block 在 Cluster 中的 Rank (0 或 1)
        // 注意：CG 没有直接提供 block_rank_in_cluster()，我们需要用 cluster API
        auto cluster = this_cluster();
        int rank = cluster.block_rank();
        
        // 写入数据：Rank 0 写 100，Rank 1 写 200
        *data = (rank + 1) * 100;
        printf("Block Rank %d: Init local SMEM = %d\n", rank, *data);
    }
    // 块内同步，确保主线程写完，其他线程看到（虽然本例只有主线程操作）
    block.sync();
}

// 本地计算（用于隐藏同步延迟）
__device__ void local_processing(const thread_block& block) {
    if (block.thread_rank() == 0) {
        // 模拟一些计算工作
        // 在实际场景中，这里可以是矩阵乘法的计算部分
        // 此时，硬件正在后台处理 Cluster 间的握手
        // printf("... Doing local work ...\n");
    }
}

// 处理远程数据
__device__ void process_shared_data(const thread_block& block, int *dsmem, int* output_debug) {
    if (block.thread_rank() == 0) {
        auto cluster = this_cluster();
        int my_rank = cluster.block_rank();
        
        // 直接读取远程指针！
        // CG 的 map_shared_rank 返回的指针可以直接解引用
        // 就像访问本地指针一样简单
        int val = *dsmem;
        
        printf("Block Rank %d: Read Neighbor's Value = %d\n", my_rank, val);
        
        // 将结果写回 Global Memory 用于验证
        output_debug[my_rank] = val;
    }
}

// ==========================================
// 2. 核心 Kernel (基于你的片段)
// ==========================================
// 必须开启 Cluster 维度支持 (SM90)
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(int* debug_out)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 900
    if (threadIdx.x == 0) printf("Error: Requires Hopper (SM90) architecture\n");
    return;
#endif

    extern __shared__ int array[];
    
    // 1. 获取 Cluster 和 Block 句柄
    auto cluster = this_cluster();
    auto block   = this_thread_block();

    // 2. 初始化本地数据
    init_shared_data(block, &array[0]);

    // 3. 非阻塞到达 (Non-blocking Arrive)
    // 告诉 Cluster 中的其他 Block：“我已经准备好了，你可以来读我了”
    // 但我不会在这里干等，我继续往下执行。
    // 这对应底层的 mbarrier.arrive
    cluster.barrier_arrive(); 

    // 4. 延迟隐藏 (Latency Hiding)
    // 在等待其他 Block 准备好的间隙，做一些本地计算
    local_processing(block);

    // 5. 【核心知识点】映射远程 Shared Memory
    // 目标：读取下一个 Rank 的数据 (Ring Pattern)
    // Rank 0 -> 读 Rank 1
    // Rank 1 -> 读 Rank 0
    unsigned int neighbor_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
    
    // map_shared_rank: 
    // 替代了之前复杂的 __cvta_generic_to_shared + set_block_rank + cast
    // 它直接返回一个指向 neighbor_rank 的 array[0] 的合法指针
    int *dsmem = cluster.map_shared_rank(&array[0], neighbor_rank);

    // 6. 阻塞等待 (Blocking Wait)
    // 确保所有其他 Block 都执行到了 barrier_arrive()
    // 这意味着它们的 Shared Memory 已经初始化完毕，可以安全读取了
    cluster.barrier_wait();

    // 7. 消费数据
    process_shared_data(block, dsmem, debug_out);

    // 8. 全局同步 (防止某些 Block 跑太快退出了，导致 Shared Memory 被回收)
    cluster.sync();
}

// ==========================================
// 3. Host 代码
// ==========================================
int main() {
    int* d_out;
    cudaMalloc(&d_out, 2 * sizeof(int));
    cudaMemset(d_out, 0, 2 * sizeof(int));

    dim3 threads(32);
    // 启动 2 个 Block 组成一个 Cluster
    dim3 blocks(2); 
    int smem_size = 1024; // 足够放 int array

    printf("Launching Cooperative Groups Cluster Kernel...\n");

    // 注意：__cluster_dims__ 已经在 Kernel 定义中指定了
    // 编译器会自动处理 Cluster 配置
    cluster_kernel<<<blocks, threads, smem_size>>>(d_out);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 验证结果
    int h_out[2];
    cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Rank 0 应该读到 Rank 1 的值 (200)
    // Rank 1 应该读到 Rank 0 的值 (100)
    printf("Verification:\n");
    printf("Rank 0 read: %d (Expected 200)\n", h_out[0]);
    printf("Rank 1 read: %d (Expected 100)\n", h_out[1]);

    if (h_out[0] == 200 && h_out[1] == 100) {
        printf("SUCCESS: Cluster CG works!\n");
    } else {
        printf("FAILED.\n");
    }

    cudaFree(d_out);
    return 0;
}