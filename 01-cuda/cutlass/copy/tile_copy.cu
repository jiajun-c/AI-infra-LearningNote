#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

// 定义数据类型为 float (4 bytes)
// 4个 float 正好 16 bytes = 128 bits，符合 uint128_t 的原子操作要求
using TA = float;

__global__ void demo_tiled_copy_kernel() {
    // ----------------------------------------------------------------
    // 1. 准备数据源 (Source Tensor)
    // ----------------------------------------------------------------
    // 定义一个 128x8   的逻辑矩阵，存放在 Shared Memory 中
    // 128 (M) * 8 (K) = 1024 个元素
    auto tensor_shape = make_shape(Int<128>{}, Int<8>{});
    __shared__ TA smem_data[128 * 8];

    // 创建 Tensor 视图 (列主序: M 维度连续)
    // 这种布局下，(0,0)=0, (1,0)=1, (2,0)=2... 方便我们观察连续性
    Tensor tensor_S = make_tensor(make_smem_ptr(smem_data), make_layout(tensor_shape, GenColMajor{}));

    // 初始化：填充线性索引 (0, 1, 2, ... 1023)
    if (thread0()) {
        for (int i = 0; i < size(tensor_S); ++i) {
            smem_data[i] = static_cast<float>(i);
        }
    }
    __syncthreads();

    // ----------------------------------------------------------------
    // 2. 定义 TiledCopy (搬运工指挥部)
    // ----------------------------------------------------------------
    // 你的定义：
    // Atom: UniversalCopy<uint128_t> -> 强制使用 128bit 向量指令 (一次搬4个float)
    // Thread Layout: 32x8 (列主序) -> 32行8列的线程阵列
    // Value Layout:  4x1  (列主序) -> 每个线程搬运 4行1列 的数据
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TA>{}, 
        Layout<Shape<_32, _8>, Stride<_1, _32>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape< _4, _1>>{}                   // Value  Layout: M-major (ColMajor)
    );

    // ----------------------------------------------------------------
    // 3. 切分 (Partition)
    // ----------------------------------------------------------------
    // 获取当前线程的切片视图 (指挥部给当前线程分配的任务)
    auto thr_copy = copyA.get_thread_slice(threadIdx.x);
    
    // 将 Source Tensor 切分，得到当前线程负责的那一小块 (Partitioned Tensor)
    // tS 的形状将是 (4, 1) -> 也就是 4 个元素
    auto tS = thr_copy.partition_S(tensor_S);

    // ----------------------------------------------------------------
    // 4. 打印结果 (Visualization)
    // ----------------------------------------------------------------
    // 只让 Thread 0 和 Thread 1 打印，避免刷屏
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=============================================\n");
        printf("   CuTe TiledCopy Visualization Demo\n");
        printf("=============================================\n");
        
        printf("\n[1] Definitions:\n");
        print("    TiledCopy: "); print(copyA); print("\n");
        print("    Source Tensor Layout: "); print(tensor_S.layout()); print("\n");

        printf("\n[2] Thread 0 View (tid=0):\n");
        printf("    Expected: Rows 0-3 of Col 0. Values: 0, 1, 2, 3\n");
        printf("    Actual Tensor Content:\n");
        print_tensor(tS); // 打印 Tensor 内容和坐标

        printf("\n[3] Thread 1 View (tid=1):\n");
        printf("    Expected: Rows 4-7 of Col 0 (Next contiguous block). Values: 4, 5, 6, 7\n");
        // 我们利用 thr_slice(1) 来模拟线程 1 看到的
        auto thr_copy_1 = copyA.get_thread_slice(1);
        auto tS_1 = thr_copy_1.partition_S(tensor_S);
        print_tensor(tS_1);

        printf("\n[4] Thread 32 View (tid=32):\n");
        printf("    Expected: Rows 0-3 of Col 1 (Next column block). Values: 128, 129, 130, 131\n");
        // Thread Layout 是 (32, 8)，ColMajor。
        // tid 0-31 是第一列 (Col 0)，tid 32-63 是第二列 (Col 1)。
        auto thr_copy_32 = copyA.get_thread_slice(32);
        auto tS_32 = thr_copy_32.partition_S(tensor_S);
        print_tensor(tS_32);
        
        printf("\n=============================================\n");
    }
}

int main() {
    // 启动 1 个 Block，256 个线程 (32x8)
    demo_tiled_copy_kernel<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}