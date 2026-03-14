#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <iomanip>
#include <cuda_runtime.h>

using namespace cute;

// =================================================================================
// 🛠️ 性能测试工具类 (PerfWrapper) - 保持不变
// =================================================================================
class VectorCopyTester {
private:
    int N;                  
    size_t bytes;           
    float *d_in, *d_out;    
    std::vector<float> h_in;
    
    // A800 PCIe 80GB 理论带宽 (H800 更高，这里仅作参考)
    const double THEORETICAL_BW_GBPS = 1935.0; 

public:
    VectorCopyTester(int num_elements) : N(num_elements) {
        bytes = N * sizeof(float);
        h_in.resize(N);
        for(int i=0; i<N; ++i) h_in[i] = static_cast<float>(i);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
        
        std::cout << "================================================================" << std::endl;
        std::cout << "🚀 Vector Copy Benchmark Initialized" << std::endl;
        std::cout << "   Data Size:    " << N / 1024 / 1024 << " M elements" << std::endl;
        std::cout << "   Data Volume:  " << (double)bytes / 1e9 << " GB (One-way)" << std::endl;
        std::cout << "   Note: TMA requires sm_90 (Hopper) architecture." << std::endl;
        std::cout << "================================================================" << std::endl;
    }

    ~VectorCopyTester() {
        cudaFree(d_in);
        cudaFree(d_out);
    }

    template <typename Func>
    void run(std::string name, Func&& kernel_launcher) {
        cudaMemset(d_out, 0, bytes);

        // Warmup
        for(int i=0; i<5; ++i) kernel_launcher(d_in, d_out, N);
        cudaDeviceSynchronize();

        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iterations = 100;
        cudaEventRecord(start);
        for(int i=0; i<iterations; ++i) {
            kernel_launcher(d_in, d_out, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double avg_ms = milliseconds / iterations;

        // Calc
        double total_rw_bytes = 2.0 * bytes; 
        double bw_gbps = (total_rw_bytes / 1e9) / (avg_ms / 1000.0);
        double efficiency = (bw_gbps / THEORETICAL_BW_GBPS) * 100.0;

        bool passed = verify();

        std::cout << std::left << std::setw(25) << name 
                  << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " | BW: " << std::setw(6) << std::setprecision(1) << bw_gbps << " GB/s"
                  << " | Eff: " << std::setw(4) << efficiency << "%"
                  << " | Check: " << (passed ? "✅ PASS" : "❌ FAIL") 
                  << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

private:
    bool verify() {
        int check_len = 1024;
        std::vector<float> h_out_head(check_len);
        std::vector<float> h_out_tail(check_len);
        cudaMemcpy(h_out_head.data(), d_out, check_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_tail.data(), d_out + (N - check_len), check_len * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < check_len; ++i) {
            if (h_out_head[i] != h_in[i]) return false;
            if (h_out_tail[i] != h_in[N - check_len + i]) return false;
        }
        return true;
    }
};

// ... (Previous Kernels: naive, cute_128bit, cute_noVec) ...
__global__ void naive_copy_kernel(float const* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

__global__ void cute_copy_kernel(float const* d_in, float* d_out, int N) {
    using namespace cute;
    auto mIn  = make_tensor(make_gmem_ptr(d_in),  make_shape(N), make_stride(Int<1>{}));
    auto mOut = make_tensor(make_gmem_ptr(d_out), make_shape(N), make_stride(Int<1>{}));
    using BlockThreads = Int<128>;
    using VecElem      = Int<4>;   
    auto copyOp = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{}, 
        Layout<Shape<BlockThreads>>{},                
        Layout<Shape<VecElem>>{}                      
    );
    auto BLOCK_TILE_SIZE = size(BlockThreads{}) * size(VecElem{});
    int blk_idx = blockIdx.x;
    if (blk_idx * BLOCK_TILE_SIZE >= N) return;
    auto gIn  = local_tile(mIn,  make_shape(BLOCK_TILE_SIZE), make_coord(blk_idx));
    auto gOut = local_tile(mOut, make_shape(BLOCK_TILE_SIZE), make_coord(blk_idx));
    auto thr_copy = copyOp.get_thread_slice(threadIdx.x);
    auto tIs = thr_copy.partition_S(gIn);
    auto tOs = thr_copy.partition_D(gOut);
    copy(copyOp, tIs, tOs);
}

// __global__ void cute_copy_kernel_no_vec(float const* d_in, float* d_out, int N) {
//     using namespace cute;
//     auto mIn  = make_tensor(make_gmem_ptr(d_in),  make_shape(N), make_stride(Int<1>{}));
//     auto mOut = make_tensor(make_gmem_ptr(d_out), make_shape(N), make_stride(Int<1>{}));
//     using BlockThreads = Int<128>;
//     using VecElem      = Int<4>;   
//     auto copyOp = make_tiled_copy(
//         Copy_Atom<UniversalCopy<float>, float>{}, // 强制 Scalar Copy
//         Layout<Shape<BlockThreads>>{},                
//         Layout<Shape<VecElem>>{}                      
//     );
//     auto BLOCK_TILE_SIZE = size(BlockThreads{}) * size(VecElem{});
//     int blk_idx = blockIdx.x;
//     if (blk_idx * BLOCK_TILE_SIZE >= N) return; 
//     auto gIn  = local_tile(mIn,  make_shape(BLOCK_TILE_SIZE), make_coord(blk_idx));
//     auto gOut = local_tile(mOut, make_shape(BLOCK_TILE_SIZE), make_coord(blk_idx));
//     auto thr_copy = copyOp.get_thread_slice(threadIdx.x);
//     auto tIs = thr_copy.partition_S(gIn);
//     auto tOs = thr_copy.partition_D(gOut);
//     copy(copyOp, tIs, tOs);
// }
// #include <cute/tensor.hpp>

__global__ void cute_copy_kernel_no_vec(float const* d_in, float* d_out, int N) {
    using namespace cute;

    // 1. 定义配置
    using BlockThreads = Int<128>; // 线程数
    using VecElem      = Int<4>;   // 向量化长度 (对应 float4)

    // 2. 计算当前 Block 的处理范围
    auto BLOCK_TILE_SIZE = size(BlockThreads{}) * size(VecElem{}); // 128 * 4 = 512
    int blk_idx = blockIdx.x;
    
    // 越界检查
    if (blk_idx * BLOCK_TILE_SIZE >= N) return;

    // 3. 【关键步骤】构建 Block 级别的 Tensor 视图
    // 我们将这段内存视为一个 (VecElem, BlockThreads) 的二维矩阵
    // Shape:  (4, 128)
    // Stride: (1, 4)  -> Column-Major，保证第0维(4个元素)在内存中是连续的
    // auto block_shape  = make_shape(VecElem{}, BlockThreads{});
    // auto block_stride = make_stride(Int<1>{}, VecElem{}); 
    auto block_shape = make_shape(BlockThreads{}, VecElem{});
    auto block_stride = make_stride(VecElem{}, Int<1>{});
    // 构建 Global Tensor (指针偏移到当前 Block)
    auto gIn  = make_tensor(make_gmem_ptr(d_in  + blk_idx * BLOCK_TILE_SIZE), 
                            make_layout(block_shape, block_stride));
    auto gOut = make_tensor(make_gmem_ptr(d_out + blk_idx * BLOCK_TILE_SIZE), 
                            make_layout(block_shape, block_stride));

    auto tIgIn  = gIn(threadIdx.x, _);
    auto tOgOut = gOut(threadIdx.x, _);
    
    copy(tIgIn, tOgOut);
}
// =================================================================================
// 🌪️ TMA Copy Kernel (Hopper Only)
// =================================================================================
// TMA 主要是 Gmem <-> Smem。所以我们必须把数据先搬到 Smem，再搬回 Gmem。
// 流程： Global(In) --(TMA Load)--> Smem --(TMA Store)--> Global(Out)
// template <class TmaLoad, class TmaStore>
// __global__ void cute_tma_copy_kernel(
//     float const* d_in, float* d_out, int N, 
//     CUTE_GRID_CONSTANT TmaLoad const tma_load,
//     CUTE_GRID_CONSTANT TmaStore const tma_store) 
// {
//     using namespace cute;
    
//     using SmemLayout = Layout<Shape<Int<128>>>; 
//     __shared__ alignas(128) float smem_buf[size(SmemLayout{})];
//     auto tSm = make_tensor(make_smem_ptr(smem_buf), SmemLayout{});

//     auto mIn  = make_tensor(make_gmem_ptr(d_in),  N, Int<1>{});
//     auto mOut = make_tensor(make_gmem_ptr(d_out), N, Int<1>{});

//     auto thr_tma_load  = tma_load.get_slice(0);
//     auto thr_tma_store = tma_store.get_slice(0);

//     int tile_idx = blockIdx.x;
//     if (tile_idx * 128 >= N) return;

//     // 【关键修复点】：去除 make_coord，直接使用 scalar index
//     auto gIn  = local_tile(mIn,  Shape<Int<128>>{}, tile_idx);
//     auto gOut = local_tile(mOut, Shape<Int<128>>{}, tile_idx);

//     Tensor tOg = thr_tma_load.partition_S(gIn);
//     Tensor tOs = thr_tma_load.partition_D(tSm);
    
//     Tensor tSs = thr_tma_store.partition_S(tSm);
//     Tensor tSg = thr_tma_store.partition_D(gOut);

//     if (threadIdx.x == 0) {
//         copy(tma_load, tOg, tOs);
//     }
    
//     cp_async_fence();
//     cp_async_wait<0>();
//     __syncthreads();

//     if (threadIdx.x == 0) {
//         copy(tma_store, tSs, tSg);
//     }
    
//     tma_store_wait<0>();
//     __syncthreads();
// }
// =================================================================================
// 🚀 Main
// =================================================================================
int main() {
    // 检查架构
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 9) {
        std::cerr << "⚠️ Error: TMA requires Hopper (sm_90) architecture. Current: sm_" 
                  << prop.major << prop.minor << std::endl;
        // 如果不是 sm_90，可以 return 或者跳过 TMA 测试
    }

    int N = 256 * 1024 * 1024;
    VectorCopyTester tester(N);

    // Baseline tests...
    tester.run("cudaMemcpy (Device2Device)", [&](const float* in, float* out, int n) {
        cudaMemcpy(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice);
    });

    tester.run("Naive CUDA (float)", [&](const float* in, float* out, int n) {
        int block = 256;
        int grid = (n + block - 1) / block;
        naive_copy_kernel<<<grid, block>>>(in, out, n);
    });

    tester.run("CuTe TiledCopy (128bit)", [&](const float* in, float* out, int n) {
        int block = 128;
        int tile = 512;
        int grid = (n + tile - 1) / tile;
        cute_copy_kernel<<<grid, block>>>(in, out, n);
    });

    tester.run("CuTe noVec TiledCopy", [&](const float* in, float* out, int n) {
        int block = 128;
        int tile = 512;
        int grid = (n + tile - 1) / tile;
        cute_copy_kernel_no_vec<<<grid, block>>>(in, out, n);
    });

    // -------------------------------------------------------------------------
    // 🌪️ TMA Test Launcher (Requires sm_90)
    // -------------------------------------------------------------------------
    // if (prop.major >= 9) {
        // tester.run("CuTe TMA Copy (G->S->G)", [&](const float* in, float* out, int n) {
        //     using namespace cute;
            
        //     // 1. 定义 TMA 所需的 Tensor Map
        //     // 【关键修改】去掉 make_stride，直接传 Int<1>{}
        //     auto mIn  = make_tensor(make_gmem_ptr(in),  n, Int<1>{});
        //     auto mOut = make_tensor(make_gmem_ptr(out), n, Int<1>{});

        //     // TMA 每次搬运的 Smem Layout
        //     auto smem_layout = Layout<Shape<Int<128>>>{}; // 128 floats

        //     // 2. 创建 TMA Load/Store 对象
        //     auto tma_load  = make_tma_copy(
        //         SM90_TMA_LOAD{},       
        //         mIn,                   
        //         smem_layout            
        //     );

        //     auto tma_store = make_tma_copy(
        //         SM90_TMA_STORE{},      
        //         mOut,                  
        //         smem_layout            
        //     );

        //     // 3. Launch
        //     int tile_size = 128; 
        //     int grid_size = (n + tile_size - 1) / tile_size;
        //     int block_size = 32; 

            // cute_tma_copy_kernel<<<grid_size, block_size>>>(in, out, n, tma_load, tma_store);
        // });
    // }

    return 0;
}