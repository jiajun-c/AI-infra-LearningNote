#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    float* a = 0;
    auto shape = make_shape(8, 35);
    auto layout = make_layout(shape);
    auto tensor = make_tensor(a, layout);
    auto tiler = make_shape(2, 7);
    auto cood = make_coord(0, 0);
    auto local = local_tile(tensor, tiler, cood);
    print(local);
}