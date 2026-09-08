#include <iostream>

using namespace std;

auto add = [](int a, int b) {
    return a + b;
};

int main() {
    int x = 10;
    int y = 100;
    int res = add(x, y);
    printf("%d\n",[ x,  y]{
        return x + y;
    }());
    
}