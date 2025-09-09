#include <iostream>
#include <memory>

int main() {
    std::allocator<int> alloc;

    int *p = alloc.allocate(10);

    for (int i = 0; i < 10; i++) {
        alloc.construct(p + i, i);
    }
    for (int i = 0; i < 10; i++)
    {
        /* code */
        std::cout << *(p + i) << std::endl;
    }

    for (int i = 0; i < 10; i++)
    {
        /* code */
        alloc.destroy(p + i);
    }
    alloc.deallocate(p, 10);
}