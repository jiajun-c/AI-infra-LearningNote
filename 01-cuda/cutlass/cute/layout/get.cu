#include "cute/layout.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), Int<4>{});
    auto s0 = get<0>(shape);
    print(s0); print("\n");

    auto s1 = get<1>(shape);
    print(s1); print("\n");
}
