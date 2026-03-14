#include <iostream>

using namespace std;

int main() {
    int x = 10;
    int y = 100;
    auto change = [=, &x]() {
        x += y;
    };
    change();
    printf("%d %d\n", x, y);
}