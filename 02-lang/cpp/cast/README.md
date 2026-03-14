# C++ 类型转换

## 1. static_cast

static_cast的用途有三个
- 基本类型转换，也可以将void*转换为对象指针
- 上行转换（将派生类指针转换为基类）
- 下行转化 (将基类指针转换为派生类指针)

其逻辑是在编译期行为，编译器会检查类型之间是否有集成，派生关系，其会调整指针的偏移量


## 2. reinterpret_cast

位层级的重解释，可以将指向这个地址的指针转换为一个完全不同类型的指针的形式，如将整数指针转换为字符指针


## 3. const_cast

移除变量的`const` 或者`volatile` 属性，

如下所示，假设我们知道这个函数其实是不会修改数据，但是其不支持传入const修饰，那么可以使用const_cast消除其const属性，如果尝试对其进行修改会出现未定义行为

```cpp
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
```
## 4. dynamic_cast

其基于安全的下行转换，用于多态类型体系，也可以进行侧向转换，在多重继承中从一个基类跳到另一个平行的基类

其逻辑基于运行时新闻和虚函数表，失败返回nullptr