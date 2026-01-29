#include "cute/layout_composed.hpp"
#include "cute/tensor_impl.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // 1. 定义一个 Layout
    // 形状: (4, 8)
    // 步长: (1, 4) -> Column-Major (列主序)，即 LayoutLeft
    auto layout = make_layout(make_shape(8, 24), LayoutLeft{});
    auto tiler = Shape<_4, _8>{};
    int *data = new int[8*24];
    for (int i = 0; i < 8*24; i++) data[i] = i;
    Tensor a = make_tensor(data, layout);
    Tensor tile_a = zipped_divide(a, tiler);
    // 2. 打印 Layout 信息

    // print(tile_a);
    auto element= tile_a(make_coord(0, 1), make_coord(0, 0));
    print(element);
    return 0;
}

// (2, 3)
// (3, 4);

// 3*24 + 4 = 72 + 4 = 76