#include "cute/layout.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    auto shape1 = make_shape(1, 3);
    auto shape2 = make_shape(2, 4);
    auto layout1 = make_layout(shape1);
    auto layout2 = make_layout(shape2);
    auto layoutA = make_layout(layout1, layout2);
    auto layoutB = append(layout1, make_layout(Int<4>{}));
    auto layoutC = prepend(layout1, make_layout(Int<5>{}));
    print(layout1);print("\n");
    print(layout2);print("\n");
    print(layoutA);print("\n");
    print(layoutB);print("\n");
    print(layoutC);print("\n");

    auto layout4dims = make_layout(make_shape(1, 2, 3, 4));
    auto groupLayout = group<0, 2>(layout4dims);
    print(groupLayout);print("\n");

// 假设你已经使用了 using namespace cute;
    auto a = make_layout(Shape <_2, Shape <_1, _6>>{},
                           Stride<_1, Stride<_6,_2>>{});

    print(a);print("\n");

    auto ca = coalesce(a, Step<_1,_1>{});
    print(ca);print("\n");

}