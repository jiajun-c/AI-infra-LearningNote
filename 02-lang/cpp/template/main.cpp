#include <cstdio>
#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

// void foo(int *a) {
//     printf("%d\n", a[1]);
// }

template <typename T>
void foo(T&& ptr) {
    if constexpr (std::is_pointer_v<std::remove_reference_t<T>>) {
        std::cout << "Success: It is a pointer!" << std::endl;
    } else {
        std::cout << "Error: It is NOT a pointer (it is an array)!" << std::endl;
    }
    printf("%d\n", ptr[1]);
}

int main() {
    int a[32];
    a[1] = 1;
    foo(a);
}