#include <iostream>

void swap(int &a, int &b) {
    int temp = b;
    b = a;
    a = temp;
}

int main() {
    int a = 10;
    int b = 1;
    swap(a, b);
    printf("%d %d\n", a, b);
}