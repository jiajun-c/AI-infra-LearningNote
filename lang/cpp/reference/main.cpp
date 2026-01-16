#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int x = 1;
    int y = x;
    int &z = y;
    printf("%x %x %x\n", &x, &y, &z);

    z = 10;
    printf("%d %d %d\n", x, y, z);
    int a = 10;
    int &&ref = 10;
    printf("%x\n", &ref);
}
