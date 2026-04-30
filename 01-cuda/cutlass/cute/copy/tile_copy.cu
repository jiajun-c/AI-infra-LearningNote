#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// Tile 大小配置
// ============================================================================
const int kTileM = 128;
const int kTileN = 128;

// ============================================================================
// Kernel 1: Naive 逐元素拷贝 (每个线程搬一个 float，作为性能基线)
// ============================================================================
__global__ void naive_copy_kernel(float* d_in, float* d_out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        d_out[idx] = d_in[idx];
    }
}

// ============================================================================
// Kernel 2: CuTe local_tile 简单拷贝 (使用 CuTe 的 copy 原语，不经过 shared memory)
// ============================================================================
__global__ void simple_local_tile_kernel(float* d_in, float* d_out, int M, int N) {
    auto global_layout = make_layout(make_shape(M, N), LayoutLeft{});
    Tensor g_in_full  = make_tensor(make_gmem_ptr(d_in),  global_layout);
    Tensor g_out_full = make_tensor(make_gmem_ptr(d_out), global_layout);

    // 每个线程负责搬一个 (4, 8) 的小 tile，共 32 个元素
    auto tiler = Shape<_4, _8>{};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = M * N / (4 * 8);
    if (tid < num_tiles) {
        auto tile_in  = local_tile(g_in_full,  tiler, tid);
        auto tile_out = local_tile(g_out_full, tiler, tid);
        copy(tile_in, tile_out);
    }
}

// ============================================================================
// Kernel 3: TiledCopy + Shared Memory 向量化拷贝
//   Global -> Shared -> Global，使用 UniversalCopy<uint128_t> 做 128-bit 向量化搬运
// ============================================================================
__global__ void tiled_copy_kernel(float* d_in, float* d_out, int M, int N) {
    // 1. 当前 Block 对应的 Tile 坐标
    int bx = blockIdx.x;
    int by = blockIdx.y;
    auto cta_coord = make_coord(bx, by);
    auto cta_shape = make_shape(Int<kTileM>{}, Int<kTileN>{});

    // 2. 构造全局 Tensor (列主序)
    auto global_layout = make_layout(make_shape(M, N), make_stride(Int<1>{}, M));
    auto g_in_full  = make_tensor(make_gmem_ptr(d_in),  global_layout);
    auto g_out_full = make_tensor(make_gmem_ptr(d_out), global_layout);

    // 3. 切分出当前 Block 的 Local Tile
    auto g_in  = local_tile(g_in_full,  cta_shape, cta_coord);
    auto g_out = local_tile(g_out_full, cta_shape, cta_coord);

    // 4. Shared Memory
    __shared__ float shm[kTileM * kTileN];
    auto s_layout = make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}));
    auto s_tensor = make_tensor(make_smem_ptr(shm), s_layout);

    // 5. Copy_Atom: 128-bit 向量化 (每次搬 4 个 float = 16 Bytes)
    using copy_op    = UniversalCopy<cute::uint128_t>;
    using copy_trait  = Copy_Traits<copy_op>;
    using copy_atom   = Copy_Atom<copy_trait, float>;

    // 6. TiledCopy 配置
    //    thr_layout: 32 个线程 (4x8)，按列主序排列
    //    val_layout: 每个线程每次搬 (4x1) 个 float（配合 128-bit）
    //    总覆盖: thr(4x8) * val(4x1) = (16, 8)，一轮覆盖 128 个元素
    //    128x128 的 Tile 需要多轮迭代（由 CuTe 自动展开）
    Layout thr_layout = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{}));
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));
    auto tiled_copy = make_tiled_copy(copy_atom{}, thr_layout, val_layout);

    // 7. Thread Slicing
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    // 8. Global -> Shared
    auto tIgI = thr_copy.partition_S(g_in);
    auto tIsI = thr_copy.partition_D(s_tensor);
    cute::copy(tiled_copy, tIgI, tIsI);
    __syncthreads();

    // 9. Shared -> Global
    auto tSsI = thr_copy.partition_S(s_tensor);
    auto tSgO = thr_copy.partition_D(g_out);
    cute::copy(tiled_copy, tSsI, tSgO);
}

// ============================================================================
// CUDA Event 计时辅助
// ============================================================================
float benchmark_kernel(std::function<void()> launch, int warmup = 5, int repeat = 20) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;  // 平均每次耗时 (ms)
}

// ============================================================================
// 正确性验证
// ============================================================================
bool verify(const std::vector<float>& h_in, float* d_out, int size) {
    std::vector<float> h_out(size);
    cudaMemcpy(h_out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        if (h_in[i] != h_out[i]) {
            std::cout << "  Mismatch at [" << i << "]: " << h_in[i] << " != " << h_out[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main: 在多个矩阵尺寸下对比三种 Kernel 的性能
// ============================================================================
int main() {
    // 测试多种矩阵大小
    std::vector<std::pair<int,int>> sizes = {
        {256,   256},
        {512,   512},
        {1024,  1024},
        {2048,  2048},
        {4096,  4096},
    };

    std::cout << "====================================================================\n";
    std::cout << "  CuTe Copy Kernel Benchmark (Global -> [Shared] -> Global)\n";
    std::cout << "====================================================================\n";
    printf("%-12s | %-14s | %-14s | %-14s | %-10s\n",
           "Size(MxN)", "Naive (ms)", "SimpleTile(ms)", "TiledCopy(ms)", "DataSize");
    std::cout << "--------------------------------------------------------------------\n";

    for (auto& [M, N] : sizes) {
        size_t num_elems = (size_t)M * N;
        size_t bytes = num_elems * sizeof(float);

        // Host 初始化
        std::vector<float> h_in(num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            h_in[i] = static_cast<float>(i % 10000);
        }

        // Device 分配
        float *d_in, *d_out;
        cudaMalloc(&d_in,  bytes);
        cudaMalloc(&d_out, bytes);
        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // ---- Kernel 1: Naive ----
        {
            int threads = 256;
            int blocks  = (num_elems + threads - 1) / threads;
            cudaMemset(d_out, 0, bytes);

            auto launch = [&]() {
                naive_copy_kernel<<<blocks, threads>>>(d_in, d_out, M, N);
            };
            float ms = benchmark_kernel(launch);

            bool ok = verify(h_in, d_out, num_elems);
            float bw = 2.0f * bytes / (ms * 1e-3) / 1e9;  // GB/s (读+写)

            printf("%-12s | %8.4f (%s) |",
                   (std::to_string(M) + "x" + std::to_string(N)).c_str(),
                   ms, ok ? "PASS" : "FAIL");
        }

        // ---- Kernel 2: Simple local_tile ----
        {
            int total_tiles = num_elems / (4 * 8);
            int threads = 256;
            int blocks  = (total_tiles + threads - 1) / threads;
            cudaMemset(d_out, 0, bytes);

            auto launch = [&]() {
                simple_local_tile_kernel<<<blocks, threads>>>(d_in, d_out, M, N);
            };
            float ms = benchmark_kernel(launch);

            bool ok = verify(h_in, d_out, num_elems);
            printf(" %8.4f (%s) |", ms, ok ? "PASS" : "FAIL");
        }

        // ---- Kernel 3: TiledCopy + SharedMemory ----
        {
            dim3 grid(M / kTileM, N / kTileN);
            dim3 block(32);  // thr_layout = (4x8) = 32 threads
            cudaMemset(d_out, 0, bytes);

            auto launch = [&]() {
                tiled_copy_kernel<<<grid, block>>>(d_in, d_out, M, N);
            };
            float ms = benchmark_kernel(launch);

            bool ok = verify(h_in, d_out, num_elems);

            // 有效带宽 (读 + 写)
            float bw = 2.0f * bytes / (ms * 1e-3) / 1e9;
            printf(" %8.4f (%s) | %.1f MB\n", ms, ok ? "PASS" : "FAIL", bytes / 1e6);
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }

    // 打印带宽汇总
    std::cout << "\n====================================================================\n";
    std::cout << "  Effective Bandwidth (GB/s) = 2 * DataSize / Time\n";
    std::cout << "====================================================================\n";
    printf("%-12s | %-14s | %-14s | %-14s\n",
           "Size(MxN)", "Naive BW", "SimpleTile BW", "TiledCopy BW");
    std::cout << "--------------------------------------------------------------------\n";

    for (auto& [M, N] : sizes) {
        size_t num_elems = (size_t)M * N;
        size_t bytes = num_elems * sizeof(float);

        float *d_in, *d_out;
        cudaMalloc(&d_in,  bytes);
        cudaMalloc(&d_out, bytes);

        std::vector<float> h_in(num_elems);
        for (size_t i = 0; i < num_elems; ++i) h_in[i] = static_cast<float>(i % 10000);
        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        float bw[3];

        // Naive
        {
            int threads = 256;
            int blocks  = (num_elems + threads - 1) / threads;
            auto launch = [&]() { naive_copy_kernel<<<blocks, threads>>>(d_in, d_out, M, N); };
            float ms = benchmark_kernel(launch);
            bw[0] = 2.0f * bytes / (ms * 1e-3) / 1e9;
        }
        // SimpleTile
        {
            int total_tiles = num_elems / (4 * 8);
            int threads = 256;
            int blocks  = (total_tiles + threads - 1) / threads;
            auto launch = [&]() { simple_local_tile_kernel<<<blocks, threads>>>(d_in, d_out, M, N); };
            float ms = benchmark_kernel(launch);
            bw[1] = 2.0f * bytes / (ms * 1e-3) / 1e9;
        }
        // TiledCopy
        {
            dim3 grid(M / kTileM, N / kTileN);
            dim3 block(32);
            auto launch = [&]() { tiled_copy_kernel<<<grid, block>>>(d_in, d_out, M, N); };
            float ms = benchmark_kernel(launch);
            bw[2] = 2.0f * bytes / (ms * 1e-3) / 1e9;
        }

        printf("%-12s | %10.2f     | %10.2f     | %10.2f\n",
               (std::to_string(M) + "x" + std::to_string(N)).c_str(),
               bw[0], bw[1], bw[2]);

        cudaFree(d_in);
        cudaFree(d_out);
    }

    std::cout << "====================================================================\n";
    return 0;
}
