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
// Kernel: CuTe Padding Version (No Swizzle, Just Padding)
// ----------------------------------------------------------------------
__global__ void transpose_cutlass_padding(float *odata, const float *idata, int width, int height)
{
    using namespace cute;

    // Tile Configuration
    using bM = Int<32>;
    using bN = Int<32>;

    // 1. Global Memory Layouts (Row-Major)
    auto gmem_layout_S = make_layout(make_shape(height, width), make_stride(width, Int<1>{}));
    auto gmem_layout_D = make_layout(make_shape(width, height), make_stride(height, Int<1>{}));

    // 2. Shared Memory Layouts with PADDING
    // 关键点：我们将 "Leading Dimension" 的 Stride 设为 33 (32 + 1)
    // 这样 32x32 的矩阵在物理内存中占据 32x33 的空间
    // 列访问的 Bank index: 0, 33%32=1, 66%32=2 ... -> 无冲突
    
    // sS: 写入视图 (Row-Major with Padding)
    // Shape: (32, 32), Stride: (33, 1)
    auto smem_layout_S = make_layout(make_shape(bM{}, bN{}), 
                                     make_stride(Int<33>{}, Int<1>{}));

    // sD: 读取视图 (Col-Major with Padding / Transposed View)
    // Shape: (32, 32), Stride: (1, 33)
    // 这里的 stride (1, 33) 对应转置后的访问模式，且指向同一块物理内存
    auto smem_layout_D = make_layout(make_shape(bM{}, bN{}), 
                                     make_stride(Int<1>{}, Int<33>{}));

    // 动态 Shared Memory
    extern __shared__ float smem_storage[];
    
    // Tensors
    Tensor S = make_tensor(make_gmem_ptr(idata), gmem_layout_S);
    Tensor D = make_tensor(make_gmem_ptr(odata), gmem_layout_D);
    Tensor gS = local_tile(S, make_shape(bM{}, bN{}), make_coord(blockIdx.y, blockIdx.x));
    Tensor gD = local_tile(D, make_shape(bN{}, bM{}), make_coord(blockIdx.x, blockIdx.y));

    Tensor sS = make_tensor(make_smem_ptr(smem_storage), smem_layout_S);
    Tensor sD = make_tensor(make_smem_ptr(smem_storage), smem_layout_D);

    // 3. Thread Layout (关键：Coalesced Access)
    // 使用 LayoutRight (Row-Major) 线程布局 (32, 8)
    // 保证 tid 0, 1, 2... 访问 Global Memory 连续地址
    auto tLayout = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});

    // 计算线性线程 ID
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Partition
    Tensor tSgS = local_partition(gS, tLayout, tid);
    Tensor tSsS = local_partition(sS, tLayout, tid);
    Tensor tDgD = local_partition(gD, tLayout, tid);
    Tensor tDsD = local_partition(sD, tLayout, tid);

    // 4. Execution
    // Copy Global -> Shared (LDG.128)
    cute::copy(tSgS, tSsS);

    cp_async_fence(); 
    cp_async_wait<0>();
    __syncthreads();

    // Copy Shared -> Global (STG.128, Conflict-Free due to Padding)
    cute::copy(tDsD, tDgD);
}

int main(int argc, char **argv)
{
    const int nx = 4096;
    const int ny = 4096; // 必须匹配，测试方阵或非方阵
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);

    // Init data
    for (int i = 0; i < nx * ny; ++i) h_idata[i] = (float)i;

    float *d_idata, *d_odata;
    CHECK(cudaMalloc(&d_idata, mem_size));
    CHECK(cudaMalloc(&d_odata, mem_size));

    CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    printf("Running CuTe Padding Version...\n");

    // ========================================================
    // 计算 Shared Memory 大小 (Padding)
    // ========================================================
    // 物理形状是 32 行 x 33 列 (33 floats per row)
    // Stride 是 33
    int smem_size_padding = 32 * 33 * sizeof(float);

    transpose_cutlass_padding<<<dimGrid, dimBlock, smem_size_padding>>>(d_odata, d_idata, nx, ny);
    
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Verify
    CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
    
    // Check first few elements (Transposed Logic: (0,1) -> (1,0) = 4096)
    // Input: 0, 1, ..., 4095, 4096...
    // Output Expect: 0, 4096, 8192...
    printf("Check: %f, %f, %f\n", h_odata[0], h_odata[1], h_odata[2]);
    
    if (h_odata[1] == nx) {
        printf("Verification Passed (Basic Check)\n");
    } else {
        printf("Verification FAILED! Expected %d, got %f\n", nx, h_odata[1]);
    }

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}