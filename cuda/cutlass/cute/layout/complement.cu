#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/util/print.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // 1. 定义全集 (The Whole)
    // 假设是一个 128 大小的空间
    auto shape_M = Int<128>{};

    // 2. 定义子集 (The Part / Tile)
    // 我们取前 32 个元素，步长为 1
    Layout layout_A = make_layout(Int<32>{}, Int<1>{});

    // 3. 计算补集 (The Rest)
    // 问：为了填满 128，我还需要怎么走？
    auto layout_R = complement(layout_A, shape_M);

    std::cout << "Whole: " << shape_M << std::endl;
    std::cout << "Part:  " << layout_A << std::endl;
    std::cout << "Rest:  " << layout_R << std::endl;
    print(layout_A(0));print("\n");
    print(layout_R(0));print("\n");
    // 预期输出 Rest: (_4):_32
    // 解释：
    // _4  -> 还需要 4 步 (因为 32 * 4 = 128)
    // _32 -> 每步跨度 32 (因为 Part 已经占了 32)
    
    
    // 4. 进阶：二维场景
    // 假设全集是 (16, 16) 列主序，也就是 Stride (1, 16)
    // 我们选了一列 (16, 1)，Stride (1, 0) <-- 注意这里的0，意味着不跨列
    // complement 会告诉我们如何跨列
    auto shape_N = make_shape(Int<16>{}, Int<16>{});
    Layout layout_B = make_layout(make_shape(_16{}, _1{}), make_stride(_1{}, _0{}));
    auto layout_BR = complement(layout_B, shape_N);
    print(layout_BR);
    return 0;
}