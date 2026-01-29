#include "cute/layout.hpp"
#include <cute/tensor.hpp>
#include <iostream>

// using namespace std;
using namespace cute;
int main() {
    auto shape = make_shape(make_shape(4, 4), 2, 3);
    auto layout = make_layout(shape);

    print(rank(layout)); print("\n");
    print(rank<0>(layout)); print("\n");
    print(rank<1>(layout)); print("\n");
}