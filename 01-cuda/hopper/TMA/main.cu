#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

using namespace cute;

// ==========================================================
// Kernel 端：严格遵循 CUTLASS TMA 测试模式
// ==========================================================
// 关键: TMA 描述符必须用 __grid_constant__ 修饰！
// TMA 硬件需要直接从 const memory 中读取 64 字节的描述符，
// 如果按普通的 by-value 传递，描述符会被拷到 local memory，
// TMA 引擎无法访问，导致 illegal memory access。
template <class TiledCopy>
__global__ void tma_hello_world(__grid_constant__ const TiledCopy tma_load, float* d_out) {
    // SMEM 缓冲区 + mbarrier
    __shared__ float smem[16 * 16];
    __shared__ alignas(16) uint64_t tma_load_mbar[1];

    // SMEM tensor（column-major，与 host 端 layout_s 一致）
    Tensor sA = make_tensor(make_smem_ptr(smem), make_layout(make_shape(_16{}, _16{})));

    // 从 TMA 描述符获取全局 tensor 的坐标视图
    Tensor mA = tma_load.get_tma_tensor(make_shape(32, 32));

    // 用 cta_tiler 做 flat_divide，把全局 tensor 切分为若干 tile
    auto cta_tiler = make_shape(_16{}, _16{});
    Tensor gA = flat_divide(mA, cta_tiler);          // (16,16, 2,2)

    // TMA partition
    auto cta_tma = tma_load.get_slice(Int<0>{});
    Tensor tAgA_x = cta_tma.partition_S(gA);         // (TMA, TMA_M, TMA_N, REST_M, REST_N)
    Tensor tAsA_x = cta_tma.partition_D(sA);         // (TMA, TMA_M, TMA_N)

    // Group 所有非 TMA 维度，方便按 stage 索引
    Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);  // (TMA, REST)
    Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);  // (TMA, REST)

    // 我们只搬第 0 个 tile（左上角 16x16 块）
    int stage = 0;

    // TMA 事务字节数 = sA 一个 tile 的大小
    constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));

    if (threadIdx.x == 0) {
        // 初始化 mbarrier
        tma_load_mbar[0] = 0;
        cute::initialize_barrier(tma_load_mbar[0], 1);
        cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);

        // 发射 TMA 异步拷贝
        copy(tma_load.with(tma_load_mbar[0]), tAgA(_, stage), tAsA(_, 0));
    }
    __syncthreads();

    // 所有线程一起等待 TMA 完成
    cute::wait_barrier(tma_load_mbar[0], 0);

    // ==========================================================
    // 验证阶段：smem → d_out
    // ==========================================================
    int idx = threadIdx.x;
    if (idx < 16 * 16) {
        d_out[idx] = smem[idx];
    }
}

// ==========================================================
// Host 端：运筹帷幄
// ==========================================================
int main() {
    constexpr int M = 32, N = 32;
    constexpr int tile_M = 16, tile_N = 16;

    // 填充 0, 1, 2, ... 1023（column-major 存储）
    std::vector<float> h_in(M * N);
    for (int i = 0; i < M * N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, M * N * sizeof(float));
    cudaMalloc(&d_out, tile_M * tile_N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 全局 tensor layout: column-major (32,32):(1,32)
    auto gmem_layout = make_layout(make_shape(M, N));
    // SMEM tile layout: column-major (16,16):(1,16)
    auto smem_layout = make_layout(make_shape(_16{}, _16{}));
    // CTA tiler: 每个 tile 覆盖 (16,16) 的区域
    auto cta_tiler = make_shape(_16{}, _16{});

    // 创建全局 tensor 并生成 TMA 描述符
    Tensor gA = make_tensor(d_in, gmem_layout);
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout, cta_tiler, Int<1>{});

    // 启动 kernel
    tma_hello_world<<<1, 256>>>(tma_load, d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 拷回验证
    std::vector<float> h_out(tile_M * tile_N);
    cudaMemcpy(h_out.data(), d_out, tile_M * tile_N * sizeof(float), cudaMemcpyDeviceToHost);

    // column-major 下，tile(0,0) 的前 16 个元素是第 0 列的前 16 行: 0,1,2,...,15
    std::cout << "TMA 成功搬运的前 16 个元素 (column-major tile 左上角):\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\n(预期: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)" << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}