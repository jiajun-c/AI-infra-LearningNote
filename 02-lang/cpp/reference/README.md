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

## 5. 引用和指针的区别

引用和指针的区别在于引用直接绑定到了变量，而指针是可以修改其所指向的地址

### 5.1 本质差异

- **引用**：是某个变量的**别名**，本质上和原变量是同一个对象，编译器通常用常量指针实现
- **指针**：是一个独立的**地址变量**，存储着另一个对象的内存地址，可以为空、可以重新指向

```cpp
int a = 10;

// 引用：a 的别名，必须初始化且不可重绑定
int& ref = a;
ref = 20;             // a 变为 20

// 指针：可以后初始化、可以为 null、可以重新指向
int* ptr = &a;
ptr = nullptr;        // 合法
int b = 30;
ptr = &b;             // 合法，重新指向 b
```

### 5.2 关键对比

| 特性 | 引用 | 指针 |
|------|------|------|
| 初始化 | 必须初始化，不能为空 | 可以为空（`nullptr`），可以后绑定 |
| 可重绑定 | 一旦绑定不可改变 | 可以随时指向其他对象 |
| 运算 | 不能做 `+/-` 算术运算 | 可以进行指针运算 |
| 独立性 | 无独立内存空间 | 占 4 / 8 字节 |
| 多级 | 只有右值引用 `&&`（语义不同） | 可有多级指针 `**` |

底层上 `int& ref = a;` 约等价于 `int* const ref = &a;`，但引用在语言层面更安全、更直观。

### 5.3 作为函数参数的选择

```cpp
// 1. const T& —— 默认选择：避免拷贝、不允许修改
void print(const vector<int>& v);

// 2. T& —— 需要修改对象
void swap(int& a, int& b);

// 3. T* —— 可能为 null / 可选项
Node* find(Node* head, int target);

// 4. T** / T*& —— 输出参数，修改指针本身
bool find(Node* head, int target, Node** pp);
```

**选择原则**：

- 默认使用 `const T&`：传非内置类型且不需要修改时，几乎都是它
- 需要修改时使用 `T&`
- 只有当参数"可能为空"或"需要重新指向"时使用指针——指针的可空性是有价值的语义信号
- 内置类型（`int`、`double` 等）和小的 POD 优先值传递

简单记：**引用是不允许 null 的指针**；**指针是被允许重新绑定的引用**。作为函数参数，引用表达"必有"，指针表达"可有可无"。

