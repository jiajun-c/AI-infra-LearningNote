#include <cuda_runtime.h>
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

template <typename T, int kM, int kN>
__global__ void downsample_direct_stride_kernel(T const* Big_ptr, T* Small_ptr) {
    // 1. 定义目标逻辑尺寸 (M/2, N/2)
    auto half_M = Int<kM / 2>{};
    auto half_N = Int<kN / 2>{};
    auto logical_shape = make_shape(half_M, half_N);

    // 2. 核心：直接定义大矩阵的降采样步长 (Stride)
    // 原始行优先步长是 (kN, 1)。隔行隔列采样后，物理步长变为原来的 2 倍。
    // 即：沿着行方向走一步，物理内存跳过 kN * 2 个元素；沿着列方向走一步，跳过 2 个元素。
    auto sampled_stride = make_stride(Int<kN * 2>{}, Int<2>{});
    
    // 生成大矩阵的“采样视图”布局
    auto layout_big_sampled = make_layout(logical_shape, sampled_stride);

    // 3. 定义小矩阵的布局 (标准的行优先连续存储)
    auto layout_small = make_layout(logical_shape, GenRowMajor{});

    // 4. 包装成 Tensor
    Tensor g_Big_Sampled = make_tensor(make_gmem_ptr(Big_ptr), layout_big_sampled);
    Tensor g_Small       = make_tensor(make_gmem_ptr(Small_ptr), layout_small);

    // 5. 划分 Tile (16x16)
    auto bS = make_shape(Int<16>{}, Int<16>{});
    auto c_tile = make_coord(blockIdx.x, blockIdx.y);

    // 切割局部 Tile
    // 虽然底层的 Stride 完全不同，但在逻辑上它们都是从 512x512 中切出 16x16
    Tensor t_big_tile   = local_tile(g_Big_Sampled, bS, c_tile);
    Tensor t_small_tile = local_tile(g_Small, bS, c_tile);

    // 6. 定义 TiledCopy 搬运器
    // 使用 256 线程 (16x16)，每个线程搬运 1 个元素
    using CopyInst = Copy_Atom<DefaultCopy, T>;
    auto tiled_copy = make_tiled_copy(
        CopyInst{},
        make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}), 
        make_layout(make_shape(Int<1>{}, Int<1>{}))                   
    );

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // 7. Partition 获取当前线程的任务
    Tensor tSgS = thr_copy.partition_S(t_big_tile);
    Tensor tDgD = thr_copy.partition_D(t_small_tile);

    // 8. 执行搬运
    copy(tiled_copy, tSgS, tDgD);
}

template <typename T, int kM, int kN>
__global__ void downsample_composition_kernel(T const* Big_ptr, T* Small_ptr) {
    auto M = Int<kM>{};
    auto N = Int<kN>{};
    auto half_M = Int<kM/2>{};
    auto half_N = Int<kN/2>{};
    auto sampled_stride = make_stride(Int<kN * 2>{}, Int<2>{});
    
    auto layout_big = make_layout(make_shape(M, N), GenRowMajor{});

    auto step_m = make_layout(make_shape(half_M), Int<2>{});
    auto step_n = make_layout(make_shape(half_N), Int<2>{});

    auto sampled_layout = composition(layout_big, make_tuple(step_m, step_n));

    auto layout_small = make_layout(make_shape(half_M, half_N), GenRowMajor{});

    Tensor g_Big_Sampled = make_tensor(make_gmem_ptr(Big_ptr), sampled_layout);
    Tensor g_Small       = make_tensor(make_gmem_ptr(Small_ptr), layout_small);

    auto bS = make_shape(Int<16>{}, Int<16>{});
    auto c_tile = make_coord(blockIdx.x, blockIdx.y);

    Tensor t_big_tile   = local_tile(g_Big_Sampled, bS, c_tile);
    Tensor t_small_tile = local_tile(g_Small, bS, c_tile);

    using CopyInst = Copy_Atom<DefaultCopy, T>;
    auto tiled_copy = make_tiled_copy(
        CopyInst{},
        make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}), // 线程排布
        make_layout(make_shape(Int<1>{}, Int<1>{}))                   // 每线程读写 1 个
    );

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    Tensor tSgS = thr_copy.partition_S(t_big_tile);
    Tensor tDgD = thr_copy.partition_D(t_small_tile);

    // 8. 执行搬运
    // 线程完全不知道它在跳跃读取，CuTe 底层算好了偏移量
    copy(tiled_copy, tSgS, tDgD);
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int half_M = M / 2;
    const int half_N = N / 2;

    size_t big_bytes = M * N * sizeof(float);
    size_t small_bytes = half_M * half_N * sizeof(float);

    float *h_big = (float*)malloc(big_bytes);
    float *h_small = (float*)malloc(small_bytes);

    // 初始化：填充线性递增数据
    for (int i = 0; i < M * N; ++i) {
        h_big[i] = static_cast<float>(i);
    }

    float *d_big, *d_small;
    (cudaMalloc(&d_big, big_bytes));
    (cudaMalloc(&d_small, small_bytes));

    (cudaMemcpy(d_big, h_big, big_bytes, cudaMemcpyHostToDevice));

    // 启动配置：Tile 是 16x16，所以 Grid 按缩小后的矩阵来算
    dim3 block(256); // 16 * 16 线程
    dim3 grid(half_M / 16, half_N / 16);

    downsample_direct_stride_kernel<float, M, N><<<grid, block>>>(d_big, d_small);
    (cudaDeviceSynchronize());

    (cudaMemcpy(h_small, d_small, small_bytes, cudaMemcpyDeviceToHost));

    // 验证结果：小矩阵的 (row, col) 应该等于 大矩阵的 (row*2, col*2)
    bool passed = true;
    for (int r = 0; r < half_M; ++r) {
        for (int c = 0; c < half_N; ++c) {
            float expected = h_big[(r * 2) * N + (c * 2)];
            float actual = h_small[r * half_N + c];
            if (expected != actual) {
                std::cout << "Mismatch at small(" << r << "," << c << "): "
                          << "expected " << expected << ", got " << actual << std::endl;
                passed = false;
                break;
            }
        }
        if (!passed) break;
    }

    if (passed) std::cout << "✓ Composition Sampling PASSED!" << std::endl;

    cudaFree(d_big); cudaFree(d_small);
    free(h_big); free(h_small);

    return 0;
}