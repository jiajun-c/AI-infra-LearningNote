#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace cute;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

// ============================================================================
// 示例 1: 基础 composition —— reshape
// 将 1D 线性数据用 2D 视角来访问，无需搬运数据
// ============================================================================
void demo_reshape() {
    printf("\n========== Demo 1: Reshape via Composition ==========\n");

    // 物理现实: 12 个元素的线性数组 [0, 1, 2, ..., 11]
    // Layout A: shape=12, stride=1
    auto A = make_layout(Int<12>{}, Int<1>{});

    // 逻辑视角: 我想看成 3x4 的列主序矩阵
    // B(i, j) = i * 1 + j * 3  →  线性偏移
    //   col0: 0, 1, 2
    //   col1: 3, 4, 5
    //   col2: 6, 7, 8
    //   col3: 9, 10, 11
    auto B = make_layout(make_shape(_3{}, _4{}), make_stride(_1{}, _3{}));

    // R = composition(A, B)
    // R(i, j) = A( B(i, j) )
    // 先用 B 将 2D 坐标映射为 1D 偏移, 再用 A 将该偏移映射到物理地址
    auto R = composition(A, B);

    print("  Physical layout A : "); print(A); print("\n");
    print("  Logical  view   B : "); print(B); print("\n");
    print("  Composed layout R : "); print(R); print("\n");

    // 验证: R(2, 1) → B(2, 1) = 2*1 + 1*3 = 5 → A(5) = 5
    printf("  R(2, 1) = %d  (expected 5)\n", int(R(2, 1)));
    // 验证: R(0, 3) → B(0, 3) = 0*1 + 3*3 = 9 → A(9) = 9
    printf("  R(0, 3) = %d  (expected 9)\n", int(R(0, 3)));
}

// ============================================================================
// 示例 2: composition 实现转置
// 对同一块数据, 用不同 stride 的视角读取 → 逻辑转置
// ============================================================================
void demo_transpose() {
    printf("\n========== Demo 2: Transpose via Composition ==========\n");

    // 原始布局: 4x3 列主序矩阵
    // B(i, j) = i * 1 + j * 4
    //   col0: 0, 1, 2, 3
    //   col1: 4, 5, 6, 7
    //   col2: 8, 9, 10, 11
    auto B = make_layout(make_shape(_4{}, _3{}), make_stride(_1{}, _4{}));

    // 转置视角: 3x4 行主序 (从原布局的角度看)
    // BT(i, j) = i * 4 + j * 1
    // 注意: shape 是 (3, 4), 行列互换了
    auto BT = make_layout(make_shape(_3{}, _4{}), make_stride(_4{}, _1{}));

    // R = composition(B, BT)
    // R(i, j) = B( BT(i, j) )
    // BT 先将转置坐标映射为原布局的线性索引
    // B 再将线性索引映射到物理偏移
    auto R = composition(B, BT);

    print("  Original  B  (4x3 ColMajor): "); print(B); print("\n");
    print("  Transpose BT (3x4 RowMajor): "); print(BT); print("\n");
    print("  Composed  R                 : "); print(R); print("\n");

    // 验证: 原矩阵 B(2, 1) = 2 + 1*4 = 6
    //       转置后 R(1, 2) 应该也是 6  (行列互换)
    printf("  B(2, 1) = %d\n", int(B(2, 1)));
    printf("  R(1, 2) = %d  (should equal B(2,1))\n", int(R(1, 2)));
}

// ============================================================================
// 示例 3: composition + Swizzle
// 将 Swizzle 变换作用在 Shared Memory 布局上, 消除 bank conflict
// 这就是你的转置 kernel 中 swizzled_layout_smem 的原理
// ============================================================================
void demo_swizzle_composition() {
    printf("\n========== Demo 3: Swizzle via Composition ==========\n");

    // 基础布局: 8x8 Row-Major (简化版，原理与 32x32 相同)
    auto base_layout = make_layout(make_shape(_8{}, _8{}), make_stride(_8{}, _1{}));

    // Swizzle 变换: Swizzle<B, M, S>
    // B=2: XOR 2 个 bit (4 行一个周期)
    // M=0: 从 bit 0 开始作为目标
    // S=3: 从 bit 3 开始取源 (即 row 的低 bit)
    auto swizzle = Swizzle<2, 0, 3>{};

    // composition(swizzle, base_layout)
    // 效果: 逻辑坐标 (row, col) → base_layout → 线性偏移 → swizzle 重排
    // swizzle 把 row 的信息 XOR 到 col 的 bit 上
    auto swizzled = composition(swizzle, base_layout);

    print("  Base layout (8x8 RowMajor) : "); print(base_layout); print("\n");
    print("  Swizzled layout            : "); print(swizzled);     print("\n");

    printf("\n  Bank mapping comparison (col=0, varying row):\n");
    printf("  %-6s %-12s %-12s %-12s\n", "row", "base_addr", "swizzle_addr", "bank_id");
    for (int row = 0; row < 8; row++) {
        int base_addr    = base_layout(row, 0);
        int swizzle_addr = swizzled(row, 0);
        int bank_base    = base_addr % 8;
        int bank_swizzle = swizzle_addr % 8;
        printf("  %-6d %-12d %-12d %-4d (was %d)\n",
               row, base_addr, swizzle_addr, bank_swizzle, bank_base);
    }
    printf("  → Without swizzle: all bank 0 (conflict!)\n");
    printf("  → With    swizzle: banks are spread out (no conflict!)\n");
}

// ============================================================================
// 示例 4: 在 GPU kernel 中使用 composition 实现矩阵转置
// Global → Smem (composition + swizzle 写入)
// Smem → Global (composition + swizzle + 转置视图读出)
// ============================================================================
template <typename T, int kM, int kN>
__global__ void transpose_with_composition(T const* __restrict__ S_ptr,
                                           T*       __restrict__ D_ptr) {
    using namespace cute;

    // ── 1. 全局布局 ──
    auto layout_S = make_layout(make_shape(Int<kM>{}, Int<kN>{}), GenRowMajor{});
    auto layout_D = make_layout(make_shape(Int<kN>{}, Int<kM>{}), GenRowMajor{});

    Tensor gS = make_tensor(make_gmem_ptr(S_ptr), layout_S);
    Tensor gD = make_tensor(make_gmem_ptr(D_ptr), layout_D);

    // ── 2. Tile 大小 ──
    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto tileShape = make_shape(bM, bN);

    // ── 3. Shared Memory 布局: composition(Swizzle, base_layout) ──
    // 这就是 composition 的核心用法:
    //   base_layout: 定义逻辑上 32x32 的数据排布
    //   Swizzle:     对物理地址做 XOR 变换
    //   composition: 将两者组合, 逻辑坐标不变, 物理地址被 swizzle 重排
    auto swizzle = Swizzle<3, 3, 3>{};

    // ★ 写入视图 (Row-Major + Swizzle)
    // composition(swizzle, RowMajor_layout)
    //   = 逻辑上按 (row, col) 访问, 物理上地址被 swizzle 打散
    auto smem_layout_S = composition(swizzle,
        make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));

    // ★ 读出视图 (Col-Major + Swizzle) → 逻辑转置
    // composition(swizzle, ColMajor_layout)
    //   = 逻辑上按 (row, col) 访问, 但 stride 交换了, 等价于转置读取
    //   同样的 swizzle 保证读出时也无 bank conflict
    auto smem_layout_D = composition(swizzle,
        make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bN)));

    __shared__ T smem_data[32 * 32];
    // 两个 Tensor 指向同一块物理内存, 只是 "视角" 不同
    Tensor sS = make_tensor(make_smem_ptr(smem_data), smem_layout_S);  // 写入视角
    Tensor sD = make_tensor(make_smem_ptr(smem_data), smem_layout_D);  // 转置读出视角

    // ── 4. Tiled Copy (8x4 线程, 每线程 1x4 元素) ──
    using CopyInst = Copy_Atom<DefaultCopy, T>;
    auto tiled_copy = make_tiled_copy(
        CopyInst{},
        make_layout(make_shape(Int<8>{}, Int<4>{}), GenRowMajor{}),
        make_layout(make_shape(Int<1>{}, Int<4>{}))
    );
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // ── 5. 分块 ──
    auto c_tile = make_coord(blockIdx.x, blockIdx.y);
    Tensor gS_tile = local_tile(gS, tileShape, c_tile);

    auto c_tile_trans = make_coord(blockIdx.y, blockIdx.x);
    Tensor gD_tile = local_tile(gD, tileShape, c_tile_trans);

    // ── 6. Partition ──
    // Global → Smem: 用写入视图 sS
    Tensor tSgS = thr_copy.partition_S(gS_tile);  // 源: global tile
    Tensor tSsS = thr_copy.partition_D(sS);        // 目标: smem 写入视图

    // Smem → Global: 用转置读出视图 sD
    Tensor tDsD = thr_copy.partition_S(sD);        // 源: smem 转置视图
    Tensor tDgD = thr_copy.partition_D(gD_tile);    // 目标: global 输出

    // ── 7. 执行 ──
    // Step 1: Global → Smem (行写入, swizzle 重排地址, 写入无 bank conflict)
    copy(tiled_copy, tSgS, tSsS);

    __syncthreads();

    // Step 2: Smem → Global (转置读出, 同样的 swizzle 保证读出无 bank conflict)
    copy(tiled_copy, tDsD, tDgD);
}

// ============================================================================
// 示例 5: composition 实现子块切片 (Subtile Slicing)
// 用 composition 从大布局中提取等间距子块
// ============================================================================
void demo_subtile_slicing() {
    printf("\n========== Demo 5: Subtile Slicing via Composition ==========\n");

    // 物理布局: 16x16 列主序
    auto A = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(_1{}, Int<16>{}));

    // 视角: 只看偶数行偶数列 → 8x8 子矩阵, stride 翻倍
    // B(i, j) = i * 2 + j * 32   (每隔一行, 每隔一列)
    auto B = make_layout(make_shape(_8{}, _8{}), make_stride(_2{}, Int<32>{}));

    auto R = composition(A, B);

    print("  Full layout A (16x16 ColMajor): "); print(A); print("\n");
    print("  Subsampling view B (8x8)      : "); print(B); print("\n");
    print("  Composed R                     : "); print(R); print("\n");

    // R(0, 0) = A(B(0,0)) = A(0) = 0
    // R(1, 0) = A(B(1,0)) = A(2) = 2   (跳过一行)
    // R(0, 1) = A(B(0,1)) = A(32) = 32 (跳过两列)
    printf("  R(0,0) = %d (expected 0)\n",  int(R(0, 0)));
    printf("  R(1,0) = %d (expected 2)\n",  int(R(1, 0)));
    printf("  R(0,1) = %d (expected 32)\n", int(R(0, 1)));
}

// ============================================================================
// CPU 参考转置
// ============================================================================
void transpose_cpu(const float* input, float* output, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            output[j * M + i] = input[i * N + j];
}

// ============================================================================
// 主函数: 运行所有 demo + GPU 验证
// ============================================================================
int main() {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   CuTe Composition 完整示例                     ║\n");
    printf("║   composition(A, B) = A ∘ B                     ║\n");
    printf("║   R(x) = A( B(x) )                              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    // ── Host 端 demo (纯 layout 计算, 无需 GPU) ──
    demo_reshape();
    demo_transpose();
    demo_swizzle_composition();
    demo_subtile_slicing();

    // ── GPU 端 demo: 用 composition + swizzle 做矩阵转置 ──
    printf("\n========== Demo 4: GPU Transpose with Composition + Swizzle ==========\n");

    constexpr int M = 4096;
    constexpr int N = 4096;
    size_t bytes = M * N * sizeof(float);

    // 分配内存
    float* h_input  = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref    = (float*)malloc(bytes);

    for (int i = 0; i < M * N; ++i) h_input[i] = (float)i;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    constexpr int tile_size = 32;
    dim3 block(32);  // 32 个线程 (8x4 TiledCopy 布局)
    dim3 grid((M + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);

    // Warmup
    transpose_with_composition<float, M, N><<<grid, block>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        transpose_with_composition<float, M, N><<<grid, block>>>(d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;
    double bandwidth = (2.0 * M * N * sizeof(float)) / (avg_ms * 1e-3) / 1e9;

    printf("  Matrix size:         %d x %d\n", M, N);
    printf("  Average time:        %.4f ms\n", avg_ms);
    printf("  Effective bandwidth: %.2f GB/s\n", bandwidth);

    // 正确性验证
    transpose_cpu(h_input, h_ref, M, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_output[i] != h_ref[i]) {
            printf("  ✗ Mismatch at %d: got %.0f, expected %.0f\n",
                   i, h_output[i], h_ref[i]);
            pass = false;
            break;
        }
    }
    if (pass) printf("  ✓ Verification PASSED!\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    free(h_ref);

    printf("\nDone.\n");
    return 0;
}
