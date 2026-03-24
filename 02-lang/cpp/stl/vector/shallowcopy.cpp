#include <iostream>
#include <vector>
#include <span>
#include <cstdio> // 修复 1：必须包含此头文件

using namespace std;

int main() {
    vector<int> v1 = {1, 2, 3, 4};
    std::span<int> view_v1(v1);
    
    // 修复 2：打印指针地址使用 %p，避免类型截断和编译器报错
    printf("v1: %p\n", (void*)v1.data());
    printf("v2: %p\n", (void*)view_v1.data());
    
    return 0;
}