#include <cstdint>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define _CG_HAS_CLUSTER_GROUP
#include <cooperative_groups.h>
using namespace std;
// -------------------------------------------------------------------------
// Part 1: 原始 DSM_CUH 的内容 (经过微调以便独立运行)
// -------------------------------------------------------------------------

namespace cg = cooperative_groups;

enum class Stage {
    LINEAR,
    ATTN,
    FFN,
    LINEAR_DEEPSEEK,
    QUK_DEEPSEEK,
    ATTN_DEEPSEEK
};

// 辅助函数：将指针转换为共享内存偏移量 (uint32_t)
__device__ __forceinline__ uint32_t smem_ptr_to_uint(void* ptr) {
    uint32_t smem_addr;
    asm("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
        : "=r"(smem_addr) : "l"(ptr));
    return smem_addr;
}

template <int cluster_size, Stage stage>
__device__ __forceinline__ void __cluster_dims__(cluster_size, 1, 1) cluster_reduce(
    const uint32_t size, const uint32_t tid, const uint32_t tile_size, 
    const uint32_t cluster_block_id, const uint32_t src_addr, const uint32_t dst_addr, 
    uint32_t barrier, uint32_t neighbor_dst_bar, half* src, half* dst
) {
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t dst_cta, neighbor_dst_addr;
    half2 buffer;
    half __align__(16) reg_input[8];

    // 初始化本地 mbarrier (线程 0 执行)
    if (tid == 0) {
        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :
            : "r"(barrier), "r"(1)
        );
    }
    cluster.sync();

    // 循环：遍历集群中的其他 Block
    // [Demo 修改]: 原始代码是 i < num_blocks - 1，这里改为 i < num_blocks 以演示完整的 All-Reduce
    for (int i = 1; i < cluster.num_blocks(); i++) {
        
        // 1. 发起异步拷贝 (线程 0 执行)
        if (tid == 0) {
            // 设置 mbarrier 期待接收的数据量 (bytes)
            asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(barrier), "r"(size)
            );

            // 计算目标 Block ID (环形通信)
            dst_cta = (cluster_block_id + i) % cluster.num_blocks();

            // 获取目标 Block 的 dst_addr 和 barrier 地址
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(barrier), "r"(dst_cta)
            );

            // 执行异步拷贝: 将本地 src 数据发送到 目标 Block 的 dst 缓冲区
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }

        // 2. 等待数据到达 (所有线程等待)
        // 等待 mbarrier 信号，表示邻居发送的数据已经到达本地 dst 缓冲区
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                        bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(barrier),
            "r"(i ^ 1) // parity flip
        );

        // 3. 计算归约 (Reduce)
        // 将接收到的 buffer (dst) 累加到本地结果 (src)
        if constexpr (stage == Stage::ATTN) {
            // 简单的 half2 向量加法
            if (tid < tile_size / 2) {
                buffer = *(half2*)(&dst[tid * 2]);
                *(half2*)(&src[tid * 2]) = __hadd2(*(half2*)(&src[tid * 2]), buffer);
            }
        } 
        // ... (省略其他 Stage 以保持 Demo 简洁) ...
        
        cluster.sync();
    }
}

// -------------------------------------------------------------------------
// Part 2: 学习用 Kernel 和 Main 函数
// -------------------------------------------------------------------------

// 定义常量
const int CLUSTER_SIZE = 2; // 使用 2 个 Block 组成的集群
const int BLOCK_DIM = 128;  // 每个 Block 128 个线程
const int TILE_SIZE = 256;  // 每个 Block 处理 256 个 half 元素

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) test_kernel(half* g_output) {
    // 1. 分配共享内存
    // 需要空间: src (结果), dst (接收缓冲), barrier
    extern __shared__ uint8_t smem_pool[];
    
    half* src = reinterpret_cast<half*>(smem_pool);
    half* dst = src + TILE_SIZE;
    uint64_t* barrier_ptr = reinterpret_cast<uint64_t*>(dst + TILE_SIZE);

    // 2. 初始化数据
    uint32_t tid = threadIdx.x;
    
    // 初始化 src: 每个元素设为 1.0
    // 初始化 dst: 清零
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        src[i] = __float2half(1.0f); 
        dst[i] = __float2half(0.0f);
    }
    
    // 获取共享内存的 uint32 偏移地址 (供 PTX 指令使用)
    uint32_t src_addr = smem_ptr_to_uint(src);
    uint32_t dst_addr = smem_ptr_to_uint(dst);
    uint32_t bar_addr = smem_ptr_to_uint(barrier_ptr);
    uint32_t data_size_bytes = TILE_SIZE * sizeof(half);

    cg::this_cluster().sync();

    // 3. 调用 Cluster Reduce
    // 使用 ATTN 模式 (简单的累加)
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        data_size_bytes, 
        tid, 
        TILE_SIZE, 
        cg::this_cluster().block_rank(), 
        src_addr, 
        dst_addr, 
        bar_addr, 
        0, // placeholder
        src, 
        dst
    );

    cg::this_cluster().sync();

    // 4. 写回结果验证
    // 只有每个 Block 的第 0 号线程写回第一个元素到 Global Memory
    if (tid == 0) {
        // 如果 Cluster Size 是 2，每个 Block 初始是 1.0，
        // 归约后应该是 1.0 + 1.0 = 2.0
        g_output[blockIdx.x] = src[0]; 
    }
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 9) {
        std::cerr << "This demo requires Hopper architecture (sm_90) or later. "
                  << "Current device: " << prop.name << " (sm_" << prop.major << prop.minor << ")" << std::endl;
        return -1;
    }

    // 设置 Grid 大小
    int num_blocks = CLUSTER_SIZE; // 只启动 1 个 Cluster
    size_t out_bytes = num_blocks * sizeof(half);
    
    half* h_out = new half[num_blocks];
    half* d_out;
    cudaMalloc(&d_out, out_bytes);

    // 计算共享内存大小
    // src (256*2) + dst (256*2) + barrier (8) = 1032 bytes
    int smem_size = (TILE_SIZE * sizeof(half)) * 2 + sizeof(uint64_t);

    // 配置 Kernel Launch 属性以启用 Cluster
    cudaLaunchConfig_t config = {0};
    config.gridDim = num_blocks;
    config.blockDim = BLOCK_DIM;
    config.dynamicSmemBytes = smem_size;
    
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CLUSTER_SIZE;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    
    config.attrs = attribute;
    config.numAttrs = 1;

    std::cout << "Launching kernel on " << prop.name << " with Cluster Size " << CLUSTER_SIZE << "..." << std::endl;

    // 启动 Kernel
    cudaError_t err = cudaLaunchKernelEx(&config, test_kernel, d_out);

    if (err != cudaSuccess) {
        std::cerr << "Kernel Launch Failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    // 验证结果
    bool pass = true;
    for (int i = 0; i < num_blocks; i++) {
        float val = __half2float(h_out[i]);
        std::cout << "Block " << i << " Result: " << val << std::endl;
        if (val != (float)CLUSTER_SIZE) {
            pass = false;
        }
    }

    if (pass) {
        std::cout << "Test PASSED! All blocks reduced successfully." << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }

    cudaFree(d_out);
    delete[] h_out;
    return 0;
}