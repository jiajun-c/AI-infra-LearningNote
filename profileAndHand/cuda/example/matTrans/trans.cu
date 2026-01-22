#include <stdio.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

#define TILE_DIM 32

// ==========================================
// 1. 朴素版本 (Naive) - 性能基准
// ==========================================
__global__ void transposeNaive(const float *idata, float *odata, int width, int height)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height)
    {
        int index_in  = y * width + x;
        int index_out = x * height + y;
        odata[index_out] = idata[index_in];
    }
}

// ==========================================
// 2. CUDA Shared Memory 原生版本
// ==========================================
__global__ void transposeCoalesced(const float *idata, float *odata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Padding 防止 Bank Conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 合并读取
    if (x < width && y < height)
    {
        int index_in = y * width + x;
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // 2. 坐标变换，准备写入
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. 合并写入 (转置在读 Shared Mem 时发生)
    if (x < height && y < width)
    {
        int index_out = y * height + x;
        odata[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// ==========================================
// 3. CuTe 优化版本 (修正了合并写入)
// ==========================================
__global__ void transposeCute(const float* idata, float* odata, int width, int height) {
    using bM = Int<TILE_DIM>;
    using bN = Int<TILE_DIM>;

    // 定义 Tensor
    // LayoutRight = RowMajor
    auto g_in  = make_tensor(make_gmem_ptr(idata), make_layout(make_shape(height, width), LayoutRight{}));
    auto g_out = make_tensor(make_gmem_ptr(odata), make_layout(make_shape(width, height), LayoutRight{}));

    // Shared Memory (Stride<33, 1>)
    __shared__ float smem[TILE_DIM * (TILE_DIM + 1)];
    auto s_tile = make_tensor(make_smem_ptr(smem), 
                              make_layout(make_shape(bM{}, bN{}), 
                                          make_stride(Int<TILE_DIM + 1>{}, Int<1>{})));

    // Tile 切分
    auto g_in_tile  = local_tile(g_in,  make_shape(bM{}, bN{}), make_coord(blockIdx.y, blockIdx.x));
    auto g_out_tile = local_tile(g_out, make_shape(bM{}, bN{}), make_coord(blockIdx.x, blockIdx.y));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // --- 读取 ---
    int gx_in = blockIdx.x * TILE_DIM + tx;
    int gy_in = blockIdx.y * TILE_DIM + ty;

    if (gy_in < height && gx_in < width) {
        s_tile(ty, tx) = g_in_tile(ty, tx);
    }

    __syncthreads();

    // --- 写入 ---
    // 关键修正：确保 g_out_tile 的第二个维度使用 tx，以保证连续内存访问 (Coalesced)
    int gx_out = blockIdx.y * TILE_DIM + tx; // Col index of Output
    int gy_out = blockIdx.x * TILE_DIM + ty; // Row index of Output

    if (gy_out < width && gx_out < height) {
        // g_out(ty, tx) -> 物理地址连续 (tx 变化快)
        // s_tile(tx, ty) -> 逻辑转置读取 (无 Bank Conflict)
        g_out_tile(ty, tx) = s_tile(tx, ty);
    }
}

// 辅助函数：验证结果
bool verify_result(const float *h_idata, const float *h_odata, int nx, int ny) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (h_odata[x * ny + y] != h_idata[y * nx + x]) {
                return false;
            }
        }
    }
    return true;
}

// 辅助函数：重置输出内存
void reset_output(float *d_odata, int size) {
    cudaMemset(d_odata, 0, size);
}

int main()
{
    // 设置矩阵大小 (尽量大一点以获得稳定耗时)
    const int nx = 4096; 
    const int ny = 4096; 
    const size_t mem_size = nx * ny * sizeof(float);

    printf("Matrix Size: %d x %d\n", nx, ny);

    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);

    for (int i = 0; i < nx * ny; ++i) h_idata[i] = i;

    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, mem_size);
    cudaMalloc(&d_odata, mem_size);

    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    // 计算 Grid 和 Block
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid((nx + TILE_DIM - 1) / TILE_DIM, (ny + TILE_DIM - 1) / TILE_DIM, 1);

    // 创建计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // ==========================================
    // 1. Warmup (预热)
    // ==========================================
    printf("Warming up GPU...\n");
    // 运行一个轻量级 Kernel 来初始化上下文
    transposeNaive<<<dimGrid, dimBlock>>>(d_idata, d_odata, nx, ny);
    cudaDeviceSynchronize();
    reset_output(d_odata, mem_size);

    // ==========================================
    // 2. 测试 Naive (朴素版)
    // ==========================================
    cudaEventRecord(start);
    transposeNaive<<<dimGrid, dimBlock>>>(d_idata, d_odata, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
    bool resNaive = verify_result(h_idata, h_odata, nx, ny);
    printf("Naive     : %8.3f ms | Bandwidth: %6.2f GB/s | Check: %s\n", 
           milliseconds, 
        //    2.0 * mem_size / (milliseconds * 1e6) * 1e9 / 1e9, // 读+写 = 2倍数据量
            (2.0 * mem_size / 1e9)/(milliseconds/1e3) ,
           resNaive ? "PASS" : "FAIL");
    
    reset_output(d_odata, mem_size);

    // ==========================================
    // 3. 测试 Shared Memory (原生优化版)
    // ==========================================
    cudaEventRecord(start);
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_idata, d_odata, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
    bool resShared = verify_result(h_idata, h_odata, nx, ny);
    printf("SharedMem : %8.3f ms | Bandwidth: %6.2f GB/s | Check: %s\n", 
           milliseconds, 
           2.0 * mem_size / (milliseconds * 1e6) * 1e9 / 1e9, 
           resShared ? "PASS" : "FAIL");

    reset_output(d_odata, mem_size);

    // ==========================================
    // 4. 测试 CuTe (模板优化版)
    // ==========================================
    cudaEventRecord(start);
    // 注意：输入在 d_idata, 输出在 d_odata
    transposeCute<<<dimGrid, dimBlock>>>(d_idata, d_odata, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
    bool resCute = verify_result(h_idata, h_odata, nx, ny);
    printf("CuTe      : %8.3f ms | Bandwidth: %6.2f GB/s | Check: %s\n", 
           milliseconds, 
           (2.0 * mem_size / (milliseconds * 1e6)) * 1e9 / 1e9, 
           resCute ? "PASS" : "FAIL");

    // 清理
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}