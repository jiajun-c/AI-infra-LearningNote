#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include <iostream>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // 1. 物理现实 (A): 一个 10个元素 的线性数组
    // Index: 0 1 2 3 4 5 6 7 8 9
    Layout A = make_layout(Int<10>{}, Int<1>{});

    // 2. 逻辑视角 (B): 我想把它看成一个 2x5 的矩阵 (列主序)
    // 意思就是: 
    // Col 0: 0, 1
    // Col 1: 2, 3 ...
    // Shape: (2, 5), Stride: (1, 2)
    Layout B = make_layout(make_shape(_2{}, _5{}), make_stride(_1{}, _2{}));

    // 3. 组合 (R = A o B)
    // R 现在就是一个 "虚拟" 的 2D 矩阵
    auto R = composition(A, B);

    print("Layout R: "); print(R); print("\n");
    // 输出: (_2,_5):(_1,_2)
    print(R(1, 2)); print("\n");

    Layout BT = make_layout(make_shape(_2{}, _5{}), make_stride(_5{}, _1{}));
    auto RT = composition(B, BT);
    print(RT(1, 2)); print("\n");

    // 4. 验证
    // 我访问 R 的 (1, 2) -> 第 2 列 第 1 行
    // 逻辑上: B(1, 2) = 1*1 + 2*2 = 5 (它是第 5 个元素)
    // 物理上: A(5) = 5
    // 结果: 5
}