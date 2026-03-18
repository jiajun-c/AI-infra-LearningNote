#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

// 检查 CUDA 错误的宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// =========================================================================
// 核心 Kernel: 使用 CuTe 视角的 Block-Level 规约
// =========================================================================
template <int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void reduce_sum_kernel(const float* int, float* out, int N) {
    auto gIn = make_tensor(make_gmem_ptr(in), make_shape(N));
    constexpr int TILE_SIZE = BLOCK_THREADS * ELEMENTS_PER_THREAD;
    auto coord = make_coord(blockIdx.x);
    auto b_in = local_tile(gIn, make_shape(Int<TILE_SIZE>{}), coord);

    auto thr_layout = make_layout(Int<BLOCK_THREADS>{});
    auto thr_in = local_partition(b_in, thr_layout, threadIdx.x);

    float thread_sum = 0.0f;
    for (int i = 0; i < size(thr_in); i++) {
        int global_idx = blockIdx.x * TILE_SIZE + i * ELEMENTS_PER_THREAD + threadIdx.x;
        if (global_idx < N) thread_sum += thr_in(i);
    }

    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    __shared__ float smem[32]; 
    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    if (laneId == 0) smem[warpID] = thread_sum;

    __syncthreads();

    if (warpID == 0) {
        float warp_sum = (lane_id < (BLOCK_THREADS / 32)) ? smem[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }

        // 7. [阶段四] 全局规约：Block 0 的 Thread 0 负责把最终结果累加到 Global Memory
        if (lane_id == 0) {
            atomicAdd(out, warp_sum);
        }
    }
}

template <int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void cute_reduce_sum_kernel(const float* in, float* out, int N) {
    
    // 1. 将全局指针包装成 CuTe Tensor (Shape: [N])
    auto gIn = make_tensor(make_gmem_ptr(in), make_shape(N));

    // 2. 切分给当前 Block (每个 Block 负责 BLOCK_THREADS * ELEMENTS_PER_THREAD 个元素)
    constexpr int TILE_SIZE = BLOCK_THREADS * ELEMENTS_PER_THREAD;
    auto coord = make_coord(blockIdx.x);
    auto b_in = local_tile(gIn, make_shape(Int<TILE_SIZE>{}), coord);

    // 3. 定义 Thread 的 Layout，并将 Block 的数据平均发给每个 Thread
    // 这样每个线程会拿到一个 Shape 为 [ELEMENTS_PER_THREAD] 的寄存器 Tensor
    auto thr_layout = make_layout(Int<BLOCK_THREADS>{});
    auto thr_in = local_partition(b_in, thr_layout, threadIdx.x);

    // 4. [阶段一] Thread 级别的局部规约
    float thread_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < size(thr_in); ++i) {
        // 关键修复：计算全局真实的 Index 用于边界保护
        // i 是当前线程处理的第几个元素，BLOCK_THREADS 是步长
        int global_idx = blockIdx.x * TILE_SIZE + i * BLOCK_THREADS + threadIdx.x;
        
        if (global_idx < N) {
            thread_sum += thr_in(i); // 直接从 CuTe Tensor 读取
        }
    }
    float thread_sum = cute::reduce(r_in, cute::sum<float>{});

    // 5. [阶段二] Warp 级别的规约 (使用 CUDA 原生 Shuffle 指令，极速交换寄存器)
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }
    // 6. [阶段三] Block 级别的规约 (使用 Shared Memory 把每个 Warp 的结果加起来)
    // 一个 Block 最多 1024 线程，即最多 32 个 Warp
    __shared__ float smem[32]; 
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // 每个 Warp 的第一个线程（lane_id == 0）包含了这个 Warp 的总和，存入共享内存
    if (lane_id == 0) {
        smem[warp_id] = thread_sum;
    }
    __syncthreads(); // 等待所有 Warp 写完

    // 让第 0 个 Warp 把 Shared Memory 里的数据再做一次规约
    if (warp_id == 0) {
        // 如果当前 Block 没有填满所有 Warp，没数据的部分补 0
        float warp_sum = (lane_id < (BLOCK_THREADS / 32)) ? smem[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }

        // 7. [阶段四] 全局规约：Block 0 的 Thread 0 负责把最终结果累加到 Global Memory
        if (lane_id == 0) {
            atomicAdd(out, warp_sum);
        }
    }
}

// =========================================================================
// Host 端封装函数
// =========================================================================
int main() {
    int N = 10000000; // 1000万个元素
    size_t bytes = N * sizeof(float);

    // 初始化 Host 数据
    std::vector<float> h_in(N, 1.0f); // 结果应该是 10000000.0f
    float h_out = 0.0f;

    // 分配 Device 内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    // 关键：全局原子加的终点必须初始化为 0
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // 配置 Kernel 参数
    constexpr int BLOCK_THREADS = 256;
    constexpr int ELEMENTS_PER_THREAD = 8;
    constexpr int TILE_SIZE = BLOCK_THREADS * ELEMENTS_PER_THREAD; // 每个 Block 吞 2048 个数
    
    // 计算需要的 Block 数量 (向上取整)
    int grid_size = (N + TILE_SIZE - 1) / TILE_SIZE;

    std::cout << "Launching Kernel with Grid: " << grid_size 
              << ", Block: " << BLOCK_THREADS << std::endl;

    // 启动 Kernel
    cute_reduce_sum_kernel<BLOCK_THREADS, ELEMENTS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝结果回 Host
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Expected Sum: " << N * 1.0f << std::endl;
    std::cout << "Actual Sum  : " << h_out << std::endl;

    // 清理
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}