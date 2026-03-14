#include <vector>
#include <iostream>

using namespace std;

int main() {
    vector<int, std::allocator<int>> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    for (int i : v) {
        cout << i << " ";
    }
    std::cout<< std::endl;
    return 0;
}