#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    // 1. 创建一个 8x8 的大矩阵，值从 0 到 63
    auto shape_8x8  = make_shape(Int<8>{}, Int<8>{});
    auto layout_8x8 = make_layout(shape_8x8, GenColMajor{}); // 列优先
    float* ptr = new float[64];
    for (int i = 0; i < 64; ++i) ptr[i] = (float)i;
    auto tensor_full = make_tensor(make_gmem_ptr(ptr), layout_8x8);

    // 2. 【第一步：local_tile】
    // 定义切片大小为 4x4
    auto tile_shape = make_shape(Int<4>{}, Int<4>{});
    // 选中坐标为 (0,0) 的那个块
    auto block_coord = make_coord(0, 0);
    auto my_tile = local_tile(tensor_full, tile_shape, block_coord);

    // 3. 【第二步：local_partition】
    // 定义线程的排布：4个线程排成 2x2 阵型
    auto thr_layout = make_layout(make_shape(Int<2>{}, Int<2>{}));
    // 假设我们现在是“线程 0”
    int thread_id = 0;
    auto my_partition = local_partition(my_tile, thr_layout, thread_id);

    // 4. 打印结果进行对比
    print("Full 8x8 Tensor (Part):\n");
    print_tensor(tensor_full); print("\n");

    print("--- After local_tile (4x4 block at 0,0) ---\n");
    print_tensor(my_tile); print("\n");

    print("--- After local_partition (Thread 0's view) ---\n");
    print("Thread 0 shape: "); print(my_partition.shape()); print("\n");
    print_tensor(my_partition); print("\n");

    delete[] ptr;
    return 0;
}