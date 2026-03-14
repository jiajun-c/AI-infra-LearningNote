#include <iostream>

template <typename T>
void foo(T x) {
    std::cout << x << std::endl;
}

int main() {
    foo(1.1);
    foo(1);
}