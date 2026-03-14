#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_base.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/print.hpp"
#include "cute/util/print_tensor.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto layout = make_layout(make_shape(Int<6>{}, Int<10>{}));
    float* ptr = new float[60];
    for (int i = 0; i < 60; i++) ptr[i] = i;
    auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
    auto tier = make_shape(Int<3>{}, Int<5>{}, Int<5>{});
    auto cta_coord = make_coord(0, 0, _);
    auto fullcoord = local_tile(tensor, tier, cta_coord, Step<_1, _1, X>{});

    auto cta_part_coord = make_coord(_, 0, _);
    auto partTensor = local_tile(tensor, tier, cta_part_coord, Step<_1, X, _1>{});
    print("full: ");print(fullcoord);print("\n");
    print("part: ");print(partTensor);print("\n");

    auto fullPartTensor = partTensor(_, _, 1, 1);
    print(fullPartTensor);print("\n");
    print_tensor(fullPartTensor);
}
