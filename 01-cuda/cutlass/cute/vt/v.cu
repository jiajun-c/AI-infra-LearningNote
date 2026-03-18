#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    // auto shape = make_shape(1, 2, 3);
    // 在编译器直接计算出结果
    auto shape = make_shape(_1{}, _2{});
    auto shape1 = make_shape(Int<1>{}, Int<2>{});

    // 在运行时计算结果
    auto shape2 = make_shape(1, 2);
    int dim = rank_v<decltype(shape)>;
    printf("dim: %d\n", dim);
}