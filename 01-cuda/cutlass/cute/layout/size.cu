#include "cute/layout.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(make_shape(4, 5), 6);
    auto layout = make_layout(shape);

    print(size<0, 0>(layout));print("\n");
}