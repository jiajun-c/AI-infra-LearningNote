# 类型操作

## typeid

使用typeid可以打印简单对象的类型，如int，double这种，如下所示，打印处i表示int
- i 代表 int
- c 代表 char
- f 代表 float
- d 代表 double

```cpp
#include <cstdio>
#include <cute/tensor.hpp>
#include <typeinfo>

// 定义一个打印类型的辅助函数
template <typename T>
__host__ __device__ void print_type_name() {
#if defined(__PRETTY_FUNCTION__)
    printf("Type is: %s\n", __PRETTY_FUNCTION__);
#elif defined(__FUNCSIG__)
    printf("Type is: %s\n", __FUNCSIG__);
#else
    printf("Type name not supported on this compiler.\n");
#endif
}

// 使用 LayoutA_Padded 定义
using namespace cute;
using LayoutA_Padded = Layout<Shape<Int<4>, Int<4>>, Stride<Int<1>, Int<5>>>;

int main() {
    // 1. 获取 cosize_v 的类型
    using TheType = decltype(cosize_v<LayoutA_Padded>);
    int x = 10;
    std::cout << "size type " << typeid((size(LayoutA_Padded{}))).name() << std::endl;
    std::cout << "x type: " << typeid(int(x)).name() << std::endl;
    // 2. 打印它
    print_type_name<TheType>();

    return 0;
}
```

## std::move

std::move其实不一定实际对数据进行拷贝，他的原理是将输入的对象转换为右值引用然后调用移动构造函数，当系统内没有对应的移动构造/拷贝函数的时候则会调用深拷贝

```cpp
#include <cstring>
#include <iostream>
#include <utility> // for std::move

class MyString {
public:
    char* data;
    int size;

    // 1. 普通构造函数
    MyString(const char* str) {
        size = strlen(str);
        data = new char[size + 1];
        memcpy(data, str, size + 1);
        std::cout << "构造函数: 分配内存" << std::endl;
    }

    // 2. 拷贝构造函数 (Copy Constructor) - 笨重
    // 触发条件：MyString b = a;
    MyString(const MyString& other) {
        size = other.size;
        data = new char[size + 1]; // 深拷贝：重新分配内存
        memcpy(data, other.data, size + 1); // 复制数据
        std::cout << "拷贝构造: 深拷贝完成" << std::endl;
    }

    // 3. 移动构造函数 (Move Constructor) - 高效！
    // 触发条件：MyString b = std::move(a);
    // 参数是 MyString&& (右值引用)
    MyString(MyString&& other) noexcept {
        // 【关键步骤 A】窃取资源 (Steal resources)
        // 直接把指针拿过来，根本不分配新内存
        this->data = other.data; 
        this->size = other.size;

        // 【关键步骤 B】置空原对象 (Null out source)
        // 必须让原来的指针失效，否则析构时会 double free
        other.data = nullptr; 
        other.size = 0;
        
        std::cout << "移动构造: 指针所有权转移 (零拷贝)" << std::endl;
    }

    // 析构函数
    ~MyString() {
        if (data != nullptr) {
            delete[] data; // 只有非空才释放
        }
    }
};

int main() {
    MyString a("Hello World"); 
    
    // 情况 1: 拷贝
    // a 是左值，匹配 Copy Constructor
    MyString b = a; 
    
    // 情况 2: 移动
    // std::move(a) 把 a 强转为右值，匹配 Move Constructor
    MyString c = std::move(a); 


    const MyString d("Hello");
    MyString e = std::move(d); //
    return 0;
}
```