#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/ptx>

// 宏：简单的 CUDA 错误检查
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// 编译命令参考:
// nvcc -o tma_test tma_test.cu -O3 -arch=sm_90 -std=c++17 -lcuda

__global__ void kernel(half* ptr, int elts)
{
    // 动态共享内存声明
    // TMA bulk copy 要求地址必须至少 16 字节对齐
    extern __shared__ __align__(16) half smem[];

    // ----------------------------------------------------------------
    // 1. 初始化 Barrier
    // ----------------------------------------------------------------
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        // 初始化 barrier，参与线程数为 blockDim.x (所有线程都需要等到数据就绪)
        init(&bar, blockDim.x);
        // 使得初始化的 barrier 对 TMA 引擎（async proxy）可见
        cde::fence_proxy_async_shared_cta();
    }
    // 同步确保 barrier 初始化完成
    __syncthreads();

    // ----------------------------------------------------------------
    // 2. Global -> Shared (异步 TMA 加载)
    // ----------------------------------------------------------------
    if (threadIdx.x == 0) {
        // cuda::memcpy_async 在 sm_90+ 且指针对齐时，底层会编译为 cp.async.bulk
        // 这里的 aligned_size_t<16> 提示编译器这一块内存大小也是 16 对齐的
        cuda::memcpy_async(
            smem, 
            ptr,
            cuda::aligned_size_t<16>(sizeof(half) * elts),
            bar
        );
    }

    // 所有线程到达 barrier (等待 transaction 计数满足)
    barrier::arrival_token token = bar.arrive();
    
    // 等待数据搬运完成
    bar.wait(std::move(token));

    // ----------------------------------------------------------------
    // 3. 计算 (SAXPY / Add One)
    // ----------------------------------------------------------------
    // 此时 smem 中的数据已经就绪
    for (int i = threadIdx.x; i < elts; i += blockDim.x) {
        // 注意：__half 的加法
        smem[i] = smem[i] + (half)1.0f;
    }

    // ----------------------------------------------------------------
    // 4. Shared -> Global (异步 TMA 写回)
    // ----------------------------------------------------------------
    
    // 4a. 内存屏障：确保刚才的计算写入 (smem[i] += 1) 对 TMA 引擎可见
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0) {
        // 发起写回指令
        cde::cp_async_bulk_shared_to_global(ptr, smem, sizeof(half) * elts);

        // 4b. 等待写回完成
        // 创建一个 bulk async-group
        cde::cp_async_bulk_commit_group();
        // 等待该组操作完成 (Wait until 0 operations form the group are pending)
        cde::cp_async_bulk_wait_group_read<0>();
    }
    
    // 注意：Shared->Global 是 Fire-and-forget 风格，
    // 如果 kernel 结束，TMA 操作也会完成。但显式 wait 是个好习惯，
    // 尤其是在 Kernel 后面还有其他 Global Memory 依赖时。
}

int main() {
    // 配置参数
    const int warps_per_block = 4;
    const int threads_per_warp = 32;
    const int elts_per_thread = 8; // 每个线程处理8个元素
    
    const int block_size = warps_per_block * threads_per_warp; // 128
    const int elts = block_size * elts_per_thread; // 1024
    const size_t bytes = sizeof(half) * elts;

    std::cout << "Elements: " << elts << ", Bytes: " << bytes << std::endl;

    // 1. Host 内存分配与初始化
    std::vector<half> h_input(elts);
    std::vector<half> h_output_gpu(elts);
    std::vector<half> h_output_cpu_ref(elts);

    // 随机数生成
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    std::cout << "Initializing data..." << std::endl;
    for(int i = 0; i < elts; i++) {
        float val = dis(gen);
        h_input[i] = (half)i;
        // 预先计算 CPU 参考结果
        h_output_cpu_ref[i] = (half)(i + 1.0f);
    }

    // 2. Device 内存分配
    half *d_ptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // 3. 启动 Kernel
    // 注意：Shared Memory 大小通过第3个参数传递
    // 必须确保 size 足够大
    std::cout << "Launching Kernel..." << std::endl;
    kernel<<<1, block_size, bytes>>>(d_ptr, elts);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 拷贝回 Host
    CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_ptr, bytes, cudaMemcpyDeviceToHost));

    // 5. 验证正确性
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    for(int i = 0; i < elts; i++) {
        float cpu_val = (float)h_output_cpu_ref[i];
        float gpu_val = (float)h_output_gpu[i];
        
        // 使用 epsilon 比较浮点数，或者直接全等（因为加1.0通常是精确的）
        if (std::abs(cpu_val - gpu_val) > 1e-3) {
            if (errors < 10) {
                std::cout << "Error at index " << i 
                          << ": CPU=" << cpu_val 
                          << ", GPU=" << gpu_val << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "PASSED! All " << elts << " elements match." << std::endl;
    } else {
        std::cout << "FAILED! Total errors: " << errors << std::endl;
    }

    // 清理
    cudaFree(d_ptr);

    return 0;
}