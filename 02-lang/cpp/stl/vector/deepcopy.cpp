#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;
    v.push_back(1);

    std::vector<int>v1 = v;
    printf("v addr %0x\n", v.data());
    printf("v1 addr %0x\n", v1.data());

    std::vector<int>v2 = {1, 2, 3, 4, 5};
    std::vector<int>v3(v1.begin(), v1.begin() + 3);
    printf("v2 addr %x\n", v2.data());
    printf("v3 addr %x\n", v3.data());

}