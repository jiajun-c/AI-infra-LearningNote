#include <iostream>

int main() {
    char buffer[512];
    int *p1, *p2, *p3;
    std::cout << "buffer addr " << (void*)buffer << std::endl;
    p1 = new int[10];
    std::cout << "p1 addr " << p1 << std::endl;

    p2 = new (buffer) int[10];
    std::cout << "p2 addr " << p2 << std::endl;

    p3 = new (buffer+10*4) int[10];
    std::cout << "p3 addr " << p3 << std::endl;
}