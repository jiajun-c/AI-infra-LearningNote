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

// =====================================================================
// Kernel 1: Naive (Baseline, 32-way Bank Conflict)
// =====================================================================
__global__ void transpose_naive(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Bank Conflict: 列访问时，所有线程访问同一个 Bank
    __shared__ float tile[TILE_DIM][TILE_DIM];

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// =====================================================================
// Kernel 2: Padding (Pure CUDA, No Conflict)
// =====================================================================
__global__ void transpose_padding(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Padding: 每一行多申请一个 float，改变 stride 为 33
    // 虽然没有位运算，但浪费了 Shared Memory 空间
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// =====================================================================
// Kernel 3: XOR Swizzle (Pure CUDA, Manual XOR, No Conflict)
// =====================================================================
// 这是 CuTe Swizzle<3,3,3> 的手动实现版本
// 不浪费空间，但引入了 XOR 计算开销
__global__ void transpose_xor_swizzle(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Load: 写入 Smem 时进行 XOR 映射
    // 逻辑坐标 (row, col) -> 物理地址 tile[row][col ^ row]
    // 这样每一行的数据被乱序存储
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = threadIdx.y + j;
        int col = threadIdx.x;
        // Swizzle Logic: Col 异或 Row
        tile[row][col ^ row] = idata[(y + j) * width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Store: 读取 Smem (此时需要读取逻辑上的转置数据)
    // 逻辑读取: tile[col][row] (相对于写入时的坐标)
    // 物理地址: tile[col][row ^ col]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = threadIdx.x;          // 原来的 col 变成了行
        int col = threadIdx.y + j;      // 原来的 row 变成了列
        
        // 我们要读取逻辑上的 tile[row][col]，对应物理上的 tile[row][col ^ row]
        odata[(y + j) * width + x] = tile[row][col ^ row];
    }
}

// =====================================================================
// Kernel 4: CuTe Optimized (Swizzle + Vectorization 128-bit)
// =====================================================================
__global__ void transpose_cutlass(float *odata, const float *idata, int width, int height)
{
    using namespace cute;
    using bM = Int<32>;
    using bN = Int<32>;

    auto gmem_layout_S = make_layout(make_shape(height, width), make_stride(width, Int<1>{}));
    auto gmem_layout_D = make_layout(make_shape(width, height), make_stride(height, Int<1>{}));

    auto swizzle = Swizzle<3, 3, 3>{};
    auto smem_layout_S = composition(swizzle, make_layout(make_shape(bM{}, bN{}), make_stride(bN{}, Int<1>{})));
    auto smem_layout_D = composition(swizzle, make_layout(make_shape(bM{}, bN{}), make_stride(Int<1>{}, bN{})));

    extern __shared__ float smem_storage[];
    
    Tensor S = make_tensor(make_gmem_ptr(idata), gmem_layout_S);
    Tensor D = make_tensor(make_gmem_ptr(odata), gmem_layout_D);
    Tensor gS = local_tile(S, make_shape(bM{}, bN{}), make_coord(blockIdx.y, blockIdx.x));
    Tensor gD = local_tile(D, make_shape(bN{}, bM{}), make_coord(blockIdx.x, blockIdx.y));

    Tensor sS = make_tensor(make_smem_ptr(smem_storage), smem_layout_S);
    Tensor sD = make_tensor(make_smem_ptr(smem_storage), smem_layout_D);

    // [关键修复]: 正确计算线性线程 ID
    // 之前使用 threadIdx.x 导致数据丢失，因为 dimBlock.y > 1
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 线程布局 (32, 8) LayoutRight -> 保证 vectorization
    auto tLayout = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});

    Tensor tSgS = local_partition(gS, tLayout, tid);
    Tensor tSsS = local_partition(sS, tLayout, tid);
    Tensor tDgD = local_partition(gD, tLayout, tid);
    Tensor tDsD = local_partition(sD, tLayout, tid);

    // 自动向量化 copy (LDG.128)
    cute::copy(tSgS, tSsS);

    cp_async_fence(); 
    cp_async_wait<0>(); 
    __syncthreads();

    // 自动向量化 copy (STG.128)
    cute::copy(tDsD, tDgD);
}

// 性能测试辅助函数
float benchmark_kernel(void (*kernel)(float*, const float*), float* d_out, float* d_in, dim3 grid, dim3 block, int shmem_size, int n_iter) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    kernel<<<grid, block, shmem_size>>>(d_out, d_in);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        kernel<<<grid, block, shmem_size>>>(d_out, d_in);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    return msec / n_iter;
}

// CuTe Kernel 重载版本适配
float benchmark_cute(float* d_out, float* d_in, int w, int h, dim3 grid, dim3 block, int shmem_size, int n_iter) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    transpose_cutlass<<<grid, block, shmem_size>>>(d_out, d_in, w, h);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        transpose_cutlass<<<grid, block, shmem_size>>>(d_out, d_in, w, h);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    return msec / n_iter;
}

int main(int argc, char **argv)
{
    const int nx = 4096;
    const int ny = 4096;
    const int mem_size = nx * ny * sizeof(float);
    const int N_ITER = 100;

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);
    for (int i = 0; i < nx * ny; ++i) h_idata[i] = (float)i;

    float *d_idata, *d_odata;
    CHECK(cudaMalloc(&d_idata, mem_size));
    CHECK(cudaMalloc(&d_odata, mem_size));
    CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    printf("Matrix Size: %d x %d, Block: %d x %d\n", nx, ny, TILE_DIM, BLOCK_ROWS);
    printf("Benchmarking %d iterations...\n\n", N_ITER);

    // 1. Naive
    float t_naive = benchmark_kernel(transpose_naive, d_odata, d_idata, dimGrid, dimBlock, 0, N_ITER);
    printf("1. Naive (Bank Conflict):   %6.3f ms\n", t_naive);

    // 2. Padding
    float t_pad = benchmark_kernel(transpose_padding, d_odata, d_idata, dimGrid, dimBlock, 0, N_ITER);
    printf("2. Padding (Pure CUDA):     %6.3f ms\n", t_pad);

    // 3. XOR Swizzle
    float t_xor = benchmark_kernel(transpose_xor_swizzle, d_odata, d_idata, dimGrid, dimBlock, 0, N_ITER);
    printf("3. XOR Swizzle (Pure CUDA): %6.3f ms\n", t_xor);

    // 4. CuTe
    int smem_size = 32 * 32 * sizeof(float);
    float t_cute = benchmark_cute(d_odata, d_idata, nx, ny, dimGrid, dimBlock, smem_size, N_ITER);
    printf("4. CuTe (Swizzle+Vec):      %6.3f ms\n", t_cute);

    printf("\n--- Comparison ---\n");
    printf("Speedup CuTe vs Naive:   %.2fx\n", t_naive / t_cute);
    printf("Speedup CuTe vs Padding: %.2fx\n", t_pad / t_cute);
    printf("Speedup CuTe vs XOR:     %.2fx\n", t_xor / t_cute);

    // Verify CuTe Correctness
    CHECK(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < nx*ny; i += 4097) { // Sample check
        int r = i / nx; int c = i % nx;
        int t_idx = c * ny + r; // Transposed index logic
        if (h_odata[t_idx] != h_idata[i]) {
            printf("Verification FAILED at index %d! Expected %f, got %f\n", t_idx, h_idata[i], h_odata[t_idx]);
            correct = false;
            break;
        }
    }
    if(correct) printf("\nVerification Passed!\n");

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}