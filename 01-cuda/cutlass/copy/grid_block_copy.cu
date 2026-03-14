#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

template <int bM, int bN>
__global__ void grid_block_copy(float* in, float* out, int M, int N) {
    auto layout = make_layout(make_shape(M, N), GenRowMajor{});
    auto gIn = make_tensor(make_gmem_ptr(in), layout);
    auto gOut = make_tensor(make_gmem_ptr(out), layout);

    auto tile_shape = make_shape(Int<bM>{}, Int<bN>{});
    auto tile_coord = make_coord(blockIdx.x, blockIdx.y);

    auto gIn_b = local_tile(gIn, tile_shape, tile_coord);
    auto gOut_b = local_tile(gOut, tile_shape, tile_coord);

    auto thr_layout = make_layout(tile_shape, LayoutRight{});
    auto tgIn = local_partition(gIn_b, thr_layout, threadIdx.x);
    auto tgOut = local_partition(gOut_b, thr_layout, threadIdx.x);

    auto rIn = make_fragment_like(tgIn);

    copy(tgIn, rIn);
    copy(rIn, tgOut);
}

template <int bM, int bN>
__global__ void grid_block_copy_ce(float* in, float* out, int M, int N) {
    auto layout = make_layout(make_shape(M, N), GenRowMajor{});
    auto gIn = make_tensor(make_gmem_ptr(in), layout);
    auto gOut = make_tensor(make_gmem_ptr(out), layout);

    auto tile_shape = make_shape(Int<bM>{}, Int<bN>{});
    auto tile_coord = make_coord(blockIdx.x, blockIdx.y);

    auto gIn_b = local_tile(gIn, tile_shape, tile_coord);
    auto gOut_b = local_tile(gOut, tile_shape, tile_coord);

    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{}, 
        Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape<_1,_4>>{}                   // Value  Layout: M-major (ColMajor)
    );
    auto thr_copy = copyA.get_slice(threadIdx.x);
    auto tgIn = thr_copy.partition_S(gIn_b);   // Source (源) 切分
    auto tgOut = thr_copy.partition_D(gOut_b); // Destination (目标) 切分

    copy(tgIn, tgOut);
}

int main() {
    // 定义矩阵尺寸 (假设是 Block 的整数倍，以简化边界检查)
    int M = 256;
    int N = 256;
    
    // 定义 Block Tile 的大小
    constexpr int bM = 16;
    constexpr int bN = 16;

    size_t bytes = M * N * sizeof(float);

    // 分配 Host 内存并初始化
    std::vector<float> h_in(M * N);
    std::vector<float> h_out(M * N, 0.0f);
    for (int i = 0; i < M * N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // 分配 Device 内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // 配置 Grid 和 Block 维度
    dim3 grid(M / bM, N / bN);
    dim3 block(bM * bN); // 每个 Block 256 个线程，每个线程搬运 1 个元素

    // 启动核函数
    grid_block_copy_ce<bM, bN><<<grid, block>>>(d_in, d_out, M, N);

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // 简单验证结果
    bool success = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_in[i] != h_out[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "CuTe copy successful!" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}