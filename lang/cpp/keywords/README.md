# C++ 关键词

## static

static 关键字用于修饰变量和函数，使得它们在编译期间就分配了内存，而不需要每次调用时都分配内存。每次调用的初始值为上一次调用的值。且
只在当前模块中可见。

### 静态函数

静态函数的作用在于调用这个函数不会访问或者修改任何非静态的数据成员

其包含的静态成员也仅会在函数被调用的第一次进行初始化，后面不能对其进行初始化，只能对其进行修改

如下所示，`static_` 该变量仅在第一次调用时进行初始化为5，然后每次`func`被调用时进行自增

```cpp
#include "../exercise.h"

// READ: `static` 关键字 <https://zh.cppreference.com/w/cpp/language/storage_duration>
// THINK: 这个函数的两个 `static` 各自的作用是什么？
static int func(int param) {
    static int static_ = param;
    // std::cout << "static_ = " << static_ << std::endl;
    return static_++;
}

int main(int argc, char **argv) {
    // TODO: 将下列 `?` 替换为正确的数字
    ASSERT(func(5) == 5, "static variable value incorrect");
    ASSERT(func(4) == 6, "static variable value incorrect");
    ASSERT(func(3) == 7, "static variable value incorrect");
    ASSERT(func(2) == 8, "static variable value incorrect");
    ASSERT(func(1) == 9, "static variable value incorrect");
    return 0;
}
```

## const

const 表示变量的值是不可变的。
