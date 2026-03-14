#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/stride.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;
// using namespace 
int main() {
    // cuda
    auto shape = make_shape(2, 4);
    auto layoutRow = make_layout(shape, LayoutRight{});
    auto layoutCol = make_layout(shape, LayoutLeft{});
    print(layoutRow);
    print(layoutCol);
}