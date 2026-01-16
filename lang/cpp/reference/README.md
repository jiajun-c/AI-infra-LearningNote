# C++ 引用

## 1. 左右值

C++ 中我们可以将元素分为两类，左值和右值，左值有名字，在内存中有固定的地址，而右值则是在表达式的右侧，是临时的变量，无法获取到其地址

## 2. 引用的进化

后续出现了左值的引用和右值的引用

(C98)左值引用可以绑定到一个变量，左值引用可以修改绑定到的变量的值

(C++11)中提出了右值引用，使得右值可以被移动拷贝到一个新的变量中

```cpp
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int x = 1;
    int y = x;
    int &z = y;
    printf("%x %x %x\n", &x, &y, &z);

    z = 10;
    printf("%d %d %d\n", x, y, z);
    int a = 10;
    int &&ref = 10;
    printf("%x\n", &ref);
}

```

## 3. 移动语义

移动语义`std::move` 本质是一个类型转换，将一个左值引用强制转换为右值引用。


