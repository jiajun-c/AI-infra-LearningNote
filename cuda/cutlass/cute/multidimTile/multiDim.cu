#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(12, 30);
    auto layout = make_layout(shape);
    float *ptr = 0;
    auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
    auto tiler = make_shape(3, 5, 6);
    auto cood = make_coord(1, _, _);
    // auto cood = make_coord(1, any, _);
    auto local = local_tile(tensor, tiler, cood, Step<_1, X, _1>{});
    print(local(_, _, 0));
}