#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int x = 1;
    int y = x;
    int &z = y;
    printf("%x %x %x\n", &x, &y, &z);
}

// 9f1b1d88 9f1b1d8c 9f1b1d8c