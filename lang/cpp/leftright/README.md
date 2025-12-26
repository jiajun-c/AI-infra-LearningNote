# C++ 左右值

## 1. 左右值

左值：是一个对象或者变量，有固定地址，可以被修改赋值

右值：不能作为左值的都为右值，往往是一个常量或者临时变量，无固定地址

## 2. 左右值表达式

表达式返回值是左值的就是左值表达式，是右值的就是右值表达式，`++i`是一个左值表达式，`i++` 是一个右值，不可以对齐进行赋值

```cpp
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;
int main() {
    int i = 1;
    ++i = i + 2;
    printf("%d\n", i);
    i = i + 2;
    printf("%d\n", i);
}

// 3 5
```

## 3. 左右值引用

左值引用本身等于另外一个变量的拷贝，右值引用等于将临时变量保存到一个变量中，后续可以对其进行修改

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
}

// 9f1b1d88 9f1b1d8c 9f1b1d8c
```
