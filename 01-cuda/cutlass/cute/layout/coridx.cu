#include <iostream>
#include <vector>
#include <cute/tensor.hpp> // 必须包含 CuTe 头文件
#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include "cute/util/print.hpp"
#include <cuda_runtime.h>

using namespace cute;

int main() {

    auto shape = make_shape(2, make_shape(4, 3));
    auto layout = make_layout(shape, LayoutRight{});
    print(layout);
    print(idx2crd(16, shape)); // (1,(1,2))
}