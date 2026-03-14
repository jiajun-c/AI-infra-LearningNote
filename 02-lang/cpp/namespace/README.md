# 命名空间

## 1. 普通命名空间

普通的命名空间可以使得各类功能分开，通过带命名空间前缀的调用来调用对应的函数

```cpp
// a.cpp
#include "b.h"

int main() {
    foo::bar();
}

// b.h
#include <iostream>

namespace foo {
  void bar();
}

// b.cpp
#include "b.h"
namespace foo
{
    void bar()
    {
        std::cout << "bar" << std::endl;
    }
} // namespace bar

```

## 2. 匿名命名空间

匿名命名空间的作用在于使得当前declaration的函数或变量只在当前文件有效，但是当多个文件中的匿名命名空间中定义了相同的函数或者变量时就会产生冲突和编译错误

匿名命名空间中不带命名空间的名称，如下所示

```cpp
namespace {
    void bar() {
        std::cout << "bar" << std::endl;
    }
}
```