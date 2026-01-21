#include <stdio.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
// 定义 Tile 尺寸，通常设为 32 (与 warp size 一致)
#define TILE_DIM 32
#define BLOCK_ROWS 8 // 每个 Block 处理的行数，用于调整占用率

#include <cute/tensor.hpp>

using namespace cute;

// 定义 Tile 大小
using bM = Int<32>;
using bN = Int<32>;

__global__ void transposeCuteSimple(const float* idata, float* odata, int width, int height) {
    // -----------------------------------------------------------
    // 1. 正确定义 Layout 和 Tensor
    // -----------------------------------------------------------
    // 错误写法: make_tensor(ptr, shape, LayoutRight{}); -> LayoutRight{} 被误作 Stride
    // 正确写法: 先用 make_layout 显式构造布局，再传给 make_tensor
    
    // 输入矩阵: (Height, Width), Row-Major
    auto layout_in = make_layout(make_shape(height, width), LayoutRight{});
    auto g_in      = make_tensor(make_gmem_ptr(idata), layout_in);

    // 输出矩阵: (Width, Height), Row-Major (注意形状是反的)
    auto layout_out = make_layout(make_shape(width, height), LayoutRight{});
    auto g_out      = make_tensor(make_gmem_ptr(odata), layout_out);

    // -----------------------------------------------------------
    // 2. Shared Memory (自动 Padding)
    // -----------------------------------------------------------
    // 定义一个 (32, 32) 的 Layout，但在物理内存上每行跨度是 33 (Padding)
    // 这样 s_tile(row, col) 会自动映射到 smem[row * 33 + col]
    __shared__ float smem[32 * 33];
    auto s_tile = make_tensor(make_smem_ptr(smem), 
                              make_layout(make_shape(bM{}, bN{}), 
                                          make_stride(Int<33>{}, Int<1>{})));

    // -----------------------------------------------------------
    // 3. 切分 Tile (Local Tile)
    // -----------------------------------------------------------
    // 当前 Block 处理的坐标 (bx, by)
    // 这里的 coord(y, x) 对应矩阵的 (行, 列) 块索引
    auto g_in_tile  = local_tile(g_in,  make_shape(bM{}, bN{}), make_coord(blockIdx.y, blockIdx.x));
    
    // 输出矩阵的块索引需要交换 (x, y)，因为我们要转置写入
    auto g_out_tile = local_tile(g_out, make_shape(bM{}, bN{}), make_coord(blockIdx.x, blockIdx.y));

    // -----------------------------------------------------------
    // 4. 执行转置
    // -----------------------------------------------------------
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // --- 读取: Global -> Shared ---
    // 为了防止矩阵边缘越界，必须检查全局坐标
    // 我们可以通过 g_in_tile 的坐标逻辑反推，或者简单地利用 coord 和 tile 大小计算
    int global_x = blockIdx.x * 32 + tx;
    int global_y = blockIdx.y * 32 + ty;

    if (global_y < height && global_x < width) {
        // g_in_tile(ty, tx) 实际上就是 g_in(global_y, global_x)
        s_tile(ty, tx) = g_in_tile(ty, tx);
    }

    __syncthreads();

    // --- 写入: Shared -> Global ---
    // 此时 global_x 和 global_y 对于输出矩阵来说含义变了
    // 输出矩阵的 shape 是 (Width, Height)
    // 我们要写入的位置是 Transposed 后的位置: Out[x][y] = In[y][x]
    
    // 重新计算输出对应的全局坐标 (基于输出矩阵的维度)
    // 注意：为了让写入合并(Coalesced)，我们希望 threadIdx.x 对应输出的最内层维度
    // 但为了保持逻辑最简单（对应你原来的逻辑），我们先按转置逻辑写：
    
    // 我们之前读入的是: smem[ty][tx] = input[gy][gx]
    // 现在我们要写出到: output[gx][gy] = smem[ty][tx]
    // 对应到 g_out_tile 内部，就是 g_out_tile(tx, ty)
    
    // 检查转置后的边界 (x < width, y < height)
    if (global_x < width && global_y < height) {
        // 读: s_tile(ty, tx) -> 也就是原来存进去的顺序
        // 写: g_out_tile(tx, ty) -> 这里的 (tx, ty) 会被 map 到 output 的 (gx, gy)
        // 从而实现转置
        g_out_tile(tx, ty) = s_tile(ty, tx);
    }
}
// ==========================================
// 1. 朴素版本 (Naive Transpose)
// 缺点：写入全局内存时内存访问不连续（非合并），带宽极低
// ==========================================
__global__ void transposeNaive(float *odata, const float *idata, int width, int height)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 边界检查
    if (x < width && y < height)
    {
        // 读：idata[y * width + x] -> 连续读取 (Coalesced)
        // 写：odata[x * height + y] -> 跨步写入 (Uncoalesced) -> 性能瓶颈
        int index_in  = y * width + x;
        int index_out = x * height + y;
        odata[index_out] = idata[index_in];
    }
}

// ==========================================
// 2. 共享内存优化版本 (Coalesced Transpose)
// 优点：利用 Shared Memory 保证读和写都是合并访问
// ==========================================
__global__ void transposeCoalesced(float *odata, const float *idata, int width, int height)
{
    // 声明共享内存
    // TILE_DIM + 1 是为了避免 Bank Conflict (存储体冲突)
    // 如果没有 +1，同一 warp 的线程访问共享内存同一列时会发生冲突
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 计算输入数据的全局坐标
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 合并读取：从 Global Memory 读到 Shared Memory
    // 每个线程读取一个元素，读取是连续的
    if (x < width && y < height)
    {
        int index_in = y * width + x;
        // 注意：这里按 [ty][tx] 存入，保持了原始布局
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // 等待 Block 内所有线程完成读取
    __syncthreads();

    // 2. 坐标变换
    // 我们要写入到输出矩阵，输出矩阵的 Block 索引通过交换 blockIdx.x 和 blockIdx.y 获得
    // 关键点：我们重新计算 x 和 y，使得 threadIdx.x 对应输出矩阵的连续地址
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. 合并写入：从 Shared Memory 写回 Global Memory
    if (x < height && y < width)
    {
        int index_out = y * height + x;
        // 读 Shared Memory 时交换坐标：tile[tx][ty]
        // 因为前面是按 [ty][tx] 存的，这里反过来读就实现了转置
        odata[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// 主函数用于测试
int main()
{
    const int nx = 2048; // 矩阵宽度
    const int ny = 2048; // 矩阵高度
    const int mem_size = nx * ny * sizeof(float);

    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);
    float *gold    = (float *)malloc(mem_size); // CPU 结果用于验证

    // 初始化数据
    for (int i = 0; i < nx * ny; ++i) h_idata[i] = i;

    // 分配设备内存
    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, mem_size);
    cudaMalloc(&d_odata, mem_size);

    // 拷贝数据到 GPU
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    // 配置 Kernel 参数
    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    
    // 如果矩阵尺寸不能被 TILE_DIM 整除，Grid 需要 +1
    if (nx % TILE_DIM) dimGrid.x++;
    if (ny % TILE_DIM) dimGrid.y++;

    printf("Executing Shared Memory Transpose...\n");
    // 启动优化 Kernel
    transposeCuteSimple<<<dimGrid, dimBlock>>>(d_idata, d_odata, nx, ny);    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // 同步并拷贝回 Host
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

    // CPU 验证 (简单的验证逻辑)
    bool correct = true;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (h_odata[x * ny + y] != h_idata[y * nx + x]) {
                printf("%f %f\n", h_odata[x * ny + y], h_idata[y * nx + x]);
                correct = false;
                break;
            }
        }
    }

    printf("%s\n", correct ? "Result PASS" : "Result FAIL");

    // 清理内存
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    free(gold);

    return 0;
}