#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // 创建一个 4x4 的矩阵视图 (M, N)
    auto tensor_shape = make_shape(Int<128>{}, Int<8>{}, Int<2>{});
    float *a = new float[size(tensor_shape)];
    auto tensor = make_tensor(make_gmem_ptr(a), tensor_shape);

    auto tiler_shape = make_shape(Int<8>{});
    auto layout = make_layout(tiler_shape);
    auto out = local_partition(tensor, layout, 0, Step<_1>{});
    print(out);
    return 0;
}