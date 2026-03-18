#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(1, 2, 3);
    printf("%d\n", rank(shape)::value);
}