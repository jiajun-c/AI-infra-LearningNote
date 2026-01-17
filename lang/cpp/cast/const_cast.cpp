#include <iostream>

// 假设这是第三方库的函数，或者旧代码
// 虽然参数是 int*（非 const），但函数内部实际上并没有修改数据
void legacyPrint(int* ptr) {
    if (ptr != nullptr) {
        std::cout << "Legacy Function Value: " << *ptr << std::endl;
    }
}

int main() {
    const int data = 100;

    // legacyPrint(&data); // ❌ 编译错误：不能把 const int* 传给 int*

    // ✅ 合法用法：
    // 我们知道 legacyPrint 不会改数据，所以暂时去除 const 属性以通过编译
    legacyPrint(const_cast<int*>(&data)); 

    return 0;
}