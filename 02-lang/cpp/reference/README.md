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

移动语义`std::move` 本质是一个类型转换，将一个左值引用强制转换为右值引用，同时我们要注意在写移动构造函数的时候需要加上expect，不然有些地方为了防止发生意外其实还是
使用移动构造函数


## 4. 万能引用和引用折叠

C++中不允许出现引用的引用，但是在模板推导中的规则是只有右值引用+右值引用才能保证其继续为右值，而其他情况下都会坍缩为左值

如果T是一个确定的类型，那么其是右值引用，其他情况下都会不一定，因为可能模板参数本身就算一个引用

除此之外，如果以一个变量的形式传递来的，那么这个变量在函数内部永远是左值

```cpp
#include <iostream>

using namespace std;

void func(int& x)  { cout << "Lvalue" << endl; }
void func(int&& x) { cout << "Rvalue" << endl; }

template<typename T>
void wrapper(T&& arg) {
    // arg 本身有一个名字叫 "arg"，所以在 wrapper 内部，arg 永远是左值！
    func(arg);
    func(std::forward<T>(arg)); 
}

int main() {
    int a = 10;
    wrapper(a);  // 传左值 -> T是int& -> arg是int& -> 调用 func(int&) (正确)
    wrapper(10); // 传右值 -> T是int  -> arg是int&& -> 但arg有名字，是左值 -> 调用 func(int&) (错误！！)

    wrapper(std::forward<int>(a));
    wrapper(std::forward<int&&>(10));

}
```