#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // 创建一个 4x4 的矩阵视图 (M, N)
    auto shape  = make_shape(Int<6>{}, Int<12>{});
    auto layout = make_layout(shape, GenRowMajor{});
    auto tensor = make_tensor(make_gmem_ptr((float*)0), layout); // 假指针

    // 定义 Tile 的大小：我们想要 2x2 的小块
    auto tile_shape = make_shape(Int<2>{}, Int<3>{});
    
    // 假设我们现在在处理第 (0, 0) 个块
    auto tile_coord = make_coord(0, 0);

    // ---------------------------------------------------------
    // 场景 A：选择全部维度 Step<_1, _1>
    // ---------------------------------------------------------
    // auto tile_both = local_tile(tensor, tile_shape, tile_coord, Step<_1, _1>{});
    // 结果 shape: (2, 2) —— 这是一个标准的 2D 小方块

    // ---------------------------------------------------------
    // 场景 B：只选行，忽略列 Step<_1, X>
    // ---------------------------------------------------------
    // auto tile_row = local_tile(tensor, tile_shape, tile_coord, Step<_1, X>{});
    // 结果 shape: (2, 4) —— 它切出了前两行，但保留了所有的列
    // 原因是：Step 里的 X 告诉它“不要在第 1 维（N）做 Tiling”

    // ---------------------------------------------------------
    // 场景 C：只选列，忽略行 Step<X, _1>
    // ---------------------------------------------------------
    auto tile_col = local_tile(tensor, tile_shape, tile_coord, Step<X, _1>{});
    // 结果 shape: (4, 2) —— 它保留了所有的行，但切出了前两列

    // std::cout << "Tile Both Shape: " << tile_both.shape() << std::endl;
    // std::cout << "Tile Row  Shape: " << tile_row.shape() << std::endl;
    std::cout << "Tile Col  Shape: " << tile_col.shape() << std::endl;

    return 0;
}