#include "cute/layout.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_zip.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    auto layout = make_layout(Int<12>{});
    auto divide_1d = logical_divide(layout, Shape<Int<3>>{});
    print(divide_1d);print("\n");

    auto layout2d = make_layout(make_shape(Int<16>{}, Int<8>{}), GenRowMajor{});
    auto divide2d = logical_divide(layout2d, make_shape(_, Int<4>{}));

    print(divide2d);print("\n");
}