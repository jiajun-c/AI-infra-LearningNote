#include "cute/numeric/int.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;
#define TILE_DIM 32
#define BLOCK_ROWS 8

// 错误检查宏
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ----------------------------------------------------------------------
// Kernel 1: Naive Transpose (存在严重 Bank Conflict)
// ----------------------------------------------------------------------
__global__ void transpose_naive(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // 声明共享内存 [32][32]
    // 映射关系：Element(row, col) 映射到 Bank (row * 32 + col) % 32
    // 如果按列访问 (固定 col, 变 row)，所有线程访问的 Bank 都是 (row*32 + c) % 32 = c
    // 即全部落在 Bank c 上 -> 32-way Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // 1. 从 Global Memory 读取到 Shared Memory (Coalesced)
    // 这里的循环是为了处理 block 内行数多于线程行数的情况
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }

    __syncthreads();

    // 计算转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // blockIdx.y 现在对应输出的 x
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // blockIdx.x 现在对应输出的 y

    // 2. 从 Shared Memory 写回 Global Memory
    // 关键点：这里读取 tile 时，使用的是 tile[threadIdx.x][threadIdx.y + j]
    // 也就是同一个 Warp 的线程 (threadIdx.x 连续变化)，访问 tile 的不同行，同一列。
    // 这会导致严重的 Bank Conflict。
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ----------------------------------------------------------------------
// Kernel 2: Swizzled Transpose (通过 Padding 消除 Conflict)
// ----------------------------------------------------------------------

// 定义 Tile 大小
using bM = Int<32>;
using bN = Int<32>;
__global__ void transpose_swizzle(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // 声明共享内存 [32][33]
    // Padding: 每一行多分配一个 float。
    // 映射关系：Element(row, col) 映射到 Bank (row * 33 + col) % 32
    // 简化： (row * 32 + row + col) % 32 = (row + col) % 32
    // 当 col 固定，row 变化时，Bank = (row + const) % 32。
    // Bank ID 会随着 row 变化而变化，不再冲突。
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// 定义 Tile 大小
using bM = Int<32>;
using bN = Int<32>;

// ----------------------------------------------------------------------
// Kernel 3: CuTe Optimized Transpose (Swizzle + Vectorized Copy)
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// Kernel 3: CuTe Optimized Transpose (Fixed Layout + Vectorization)
// ----------------------------------------------------------------------
__global__ void transpose_cutlass(float *odata, const float *idata, int width, int height)
{
    using namespace cute;

    // =========================================================
    // 1. 定义布局与张量 (Layouts & Tensors)
    // =========================================================
    // 这里的 bM, bN 对应 Tile 大小 (32, 32)
    using bM = Int<32>;
    using bN = Int<32>;

    // 全局布局 (Row-Major)
    auto gmem_layout_S = make_layout(make_shape(height, width), make_stride(width, Int<1>{}));
    auto gmem_layout_D = make_layout(make_shape(width, height), make_stride(height, Int<1>{}));

    // Shared Memory 布局 (Swizzle)
    auto swizzle = Swizzle<3, 3, 3>{};
    // sS: 写入视图 (Row-Major + Swizzle)
    auto smem_layout_S = composition(swizzle, 
                                     make_layout(make_shape(bM{}, bN{}), 
                                                 make_stride(bN{}, Int<1>{})));
    // sD: 读取视图 (Col-Major + Swizzle) -> 逻辑上的转置
    auto smem_layout_D = composition(swizzle, 
                                     make_layout(make_shape(bM{}, bN{}), 
                                                 make_stride(Int<1>{}, bN{})));

    // 动态 Shared Memory
    __shared__ float smem_storage[1024];
    
    // 构造 Tensor
    Tensor S = make_tensor(make_gmem_ptr(idata), gmem_layout_S);
    Tensor D = make_tensor(make_gmem_ptr(odata), gmem_layout_D);
    Tensor gS = local_tile(S, make_shape(bM{}, bN{}), make_coord(blockIdx.y, blockIdx.x));
    Tensor gD = local_tile(D, make_shape(bN{}, bM{}), make_coord(blockIdx.x, blockIdx.y));

    Tensor sS = make_tensor(make_smem_ptr(smem_storage), smem_layout_S);
    Tensor sD = make_tensor(make_smem_ptr(smem_storage), smem_layout_D);

    // =========================================================
    // 2. [核心] 使用 local_partition 分发任务
    // =========================================================
    
    // 定义线程布局 (Thread Layout):
    // 对应 dimBlock(32, 8) = 256 线程
    // LayoutRight (行主序) 意味着相邻的线程 ID (0, 1) 在 N 维度 (列) 上连续
    // 这与 Global Memory 的 Row-Major 布局完美匹配 -> 合并访问 (Coalescing)
    auto tLayout = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});

    // 将数据切分给每个线程
    // 逻辑：Tile(32x32) / Thread(32x8) = 每个线程负责 (1x4) 个元素
    // tSgS: 线程负责的 Global Input 片段
    // tSsS: 线程负责的 Shared Input 片段
    Tensor tSgS = local_partition(gS, tLayout, threadIdx.x);
    Tensor tSsS = local_partition(sS, tLayout, threadIdx.x);
    
    Tensor tDgD = local_partition(gD, tLayout, threadIdx.x);
    Tensor tDsD = local_partition(sD, tLayout, threadIdx.x);

    // =========================================================
    // 3. 执行 Copy (自动向量化)
    // =========================================================

    // Step 1: Global -> Shared
    // 由于 tSgS 是 LayoutRight, tSsS 也是 LayoutRight
    // 且每个线程负责连续的 4 个 float
    // cute::copy 会自动编译为 LDG.128 (128-bit 向量加载)
    cute::copy(tSgS, tSsS);

    cp_async_fence(); 
    cp_async_wait<0>(); // 确保拷贝完成 (如果是普通 copy，这行通常会被优化掉，但加上无害)
    __syncthreads();

    // Step 2: Shared -> Global
    // 从 sD (转置视图) 读取，写入 gD
    // gD 是 LayoutRight, 写入时是合并的 (STG.128)
    // sD 读取时虽然不是连续地址 (stride=32)，但因为有 Swizzle，所以无 Bank Conflict
    cute::copy(tDsD, tDgD);
}

int main(int argc, char **argv)
{
    const int nx = 2048;
    const int ny = 2048;
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);
    float *gold    = (float *)malloc(mem_size);

    for (int i = 0; i < nx * ny; ++i) h_idata[i] = (float)i;

    float *d_idata, *d_odata;
    CHECK(cudaMalloc(&d_idata, mem_size));
    CHECK(cudaMalloc(&d_odata, mem_size));

    CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // Warmup
    transpose_naive<<<dimGrid, dimBlock>>>(d_odata, d_idata);

    // Record Naive
    printf("Running Naive...\n");
    transpose_naive<<<dimGrid, dimBlock>>>(d_odata, d_idata);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Record Swizzle
    printf("Running Swizzle (Padding)...\n");
    transpose_swizzle<<<dimGrid, dimBlock>>>(d_odata, d_idata);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
    
    // Simple verification
    // ... (omitted for brevity)
    transpose_cutlass<<<dimGrid, dimBlock>>>(d_odata, d_idata, nx, ny);

    printf("Running cutlass...\n");
    transpose_cutlass<<<dimGrid, dimBlock>>>(d_odata, d_idata, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 8; ++i) {
        printf("%f ", h_idata[i]);
        // if (h_odata[i] != gold[i]) {
        //     printf("Error at %d: %f != %f\n", i, h_odata[i], gold[i]);
        //     return -1;
        // }
    }
    printf("\n");
    CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) {
        printf("%f ", h_odata[i]);
        // if (h_odata[i] != gold[i]) {
        //     printf("Error at %d: %f != %f\n", i, h_odata[i], gold[i]);
        //     return -1;
        // }
    }
    cudaFree(d_idata);
    cudaFree(d_odata);
    CHECK(cudaGetLastError());
    free(h_idata);
    free(h_odata);
    free(gold);

    return 0;
}