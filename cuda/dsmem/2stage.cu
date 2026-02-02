#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;

// 定义计算阶段
enum class Stage {
    LINEAR, // 纯累加
    FFN     // 累加 + ReLU
};

// ==========================================
// 1. DSMEM 版本 (Cluster Optimized)
// ==========================================
template <Stage stage>
__global__ void __cluster_dims__(2, 1, 1) cluster_reduce_dsmem(half* input, half* output, int size) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    extern __shared__ half smem[];

    int my_rank = cluster.block_rank();
    int tid = block.thread_rank();
    int num_threads = block.num_threads();

    // 模拟：数据已经在 Shared Memory 中 (Load from Global)
    for (int i = tid; i < size; i += num_threads) {
        smem[i] = input[my_rank * size + i];
    }
    cluster.sync(); 

    // DSMEM 映射
    int neighbor_rank = (my_rank + 1) % cluster.num_blocks();
    half* neighbor_smem = cluster.map_shared_rank(&smem[0], neighbor_rank);

    // 计算：访问 DSMEM
    for (int i = tid; i < size; i += num_threads) {
        half val_local = smem[i];
        half val_remote = neighbor_smem[i]; // <--- 关键点：走 Cluster Network

        if constexpr (stage == Stage::LINEAR) {
            smem[i] = __hadd(val_local, val_remote);
        } else if constexpr (stage == Stage::FFN) {
            half sum = __hadd(val_local, val_remote);
            half zero = __float2half(0.0f);
            smem[i] = __hgt(sum, zero) ? sum : zero;
        }
    }
    cluster.sync();

    // Store
    for (int i = tid; i < size; i += num_threads) {
        output[my_rank * size + i] = smem[i];
    }
}

// ==========================================
// 2. 普通版本 (No DSMEM / Global Memory Fallback)
// ==========================================
// 模拟：如果无法直接访问邻居 SMEM，必须走 Global Memory
template <Stage stage>
__global__ void __cluster_dims__(2, 1, 1) cluster_reduce_baseline(half* input, half* output, int size) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    extern __shared__ half smem[];

    int my_rank = cluster.block_rank();
    int neighbor_rank = (my_rank + 1) % cluster.num_blocks();
    int tid = block.thread_rank();
    int num_threads = block.num_threads();

    // 模拟：数据加载到 SMEM
    for (int i = tid; i < size; i += num_threads) {
        smem[i] = input[my_rank * size + i];
    }
    cluster.sync(); 

    // 计算：访问 Global Memory (HBM)
    for (int i = tid; i < size; i += num_threads) {
        half val_local = smem[i];
        
        // <--- 关键点：必须回 HBM 去读邻居的数据
        // 假设邻居已经把数据准备好在 Input 中 (模拟最理想情况)
        // 在真实场景中，邻居可能还需要先 Write HBM，这里我们只计算 Read HBM 的开销
        half val_remote = input[neighbor_rank * size + i]; 

        if constexpr (stage == Stage::LINEAR) {
            smem[i] = __hadd(val_local, val_remote);
        } else if constexpr (stage == Stage::FFN) {
            half sum = __hadd(val_local, val_remote);
            half zero = __float2half(0.0f);
            smem[i] = __hgt(sum, zero) ? sum : zero;
        }
    }
    cluster.sync();

    // Store
    for (int i = tid; i < size; i += num_threads) {
        output[my_rank * size + i] = smem[i];
    }
}

// ==========================================
// 性能测试工具
// ==========================================
void benchmark(const char* name, void(*kernel)(half*, half*, int), 
               half* d_in, half* d_out, int size, int smem_size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(256);
    dim3 blocks(2); // Cluster Size 2

    // Warmup
    for(int i=0; i<10; i++) {
        kernel<<<blocks, threads, smem_size>>>(d_in, d_out, size);
    }
    cudaDeviceSynchronize();

    // Timing Loop
    int iterations = 1000;
    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) {
        kernel<<<blocks, threads, smem_size>>>(d_in, d_out, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("[%s] Avg Time: %.4f us\n", name, (milliseconds / iterations) * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ==========================================
// Host 端代码
// ==========================================
int main() {
    // 增加数据量以凸显带宽差异 (16KB per block, close to max shared per kernel without config)
    int size = 8192*2; 
    int total_elements = size * 2;
    size_t bytes = total_elements * sizeof(half);

    std::vector<half> h_in(total_elements);
    for (int i = 0; i < total_elements; ++i) h_in[i] = __float2half(1.0f);

    half *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    int smem_size = size * sizeof(half);
    
    printf("Benchmark Configuration:\n");
    printf("  Elements per Block: %d (FP16)\n", size);
    printf("  SMEM Usage: %d KB\n", smem_size / 1024);
    printf("  Hardware: Requires H100 (SM90)\n\n");

    // 1. 运行基准 (Global Memory Access)
    benchmark("Baseline (Global Mem)", 
              cluster_reduce_baseline<Stage::LINEAR>, 
              d_in, d_out, size, smem_size);

    // 2. 运行 DSMEM (Cluster Access)
    benchmark("DSMEM (Cluster Mem)", 
              cluster_reduce_dsmem<Stage::LINEAR>, 
              d_in, d_out, size, smem_size);

    // 验证结果一致性
    std::vector<half> h_out_dsmem(total_elements);
    std::vector<half> h_out_base(total_elements);
    
    cluster_reduce_dsmem<Stage::LINEAR><<<2, 256, smem_size>>>(d_in, d_out, size);
    cudaMemcpy(h_out_dsmem.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cluster_reduce_baseline<Stage::LINEAR><<<2, 256, smem_size>>>(d_in, d_out, size);
    cudaMemcpy(h_out_base.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for(int i=0; i<total_elements; i++) {
        float v1 = __half2float(h_out_dsmem[i]);
        float v2 = __half2float(h_out_base[i]);
        if (abs(v1 - v2) > 0.01) {
            correct = false;
            break;
        }
    }
    printf("\nVerification: %s\n", correct ? "PASSED" : "FAILED");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}