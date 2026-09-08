# C++ lambda 表达式

## 1. 基本语法

```cpp
  [capture](parameters) -> return_type {
      body
  };
```

如下所示

```cpp
#include <iostream>

using namespace std;

auto add = [](int a, int b) {
    return a + b;
};

int main() {
    int x = 10;
    int y = 100;
    int res = add(x, y);
    printf("%d\n",[ x,  y]{
        return x + y;
    }());
    
}
```

## 2 捕获子句

- [] 不捕获
- [=] 按值捕获所有用到的值
- [&] 按引用捕获所有用到的值
- [x] 仅按值捕获所有用到的值
- [&x] 仅按引用捕获x
- [this] 捕获当前对象的this指针

例子

```cpp
std::sort(vec.begin(), vec.end(), [](int a, int b){
    return a > b;
})
```

`[this]` 捕获当前对象的指针，lambda 内可直接访问成员变量。

```cpp
class Worker {
    std::string name_;
public:
    Worker(std::string name) : name_(std::move(name)) {}

    void hello() {
        // [this] 捕获当前对象的指针，name_ 实际是 this->name_
        auto f = [this]() {
            std::cout << "Hi, " << name_ << std::endl;
        };
        f();
    }
};

int main() {
    Worker{"Alice"}.hello();   // 输出：Hi, Alice
}
```

⚠️ 注意：`[this]` 持有的是裸指针，**不会延长对象生命周期**。下面这种写法就是 UB：

```cpp
auto f = Worker{"Alice"}.hello();   // 临时对象在表达式结束后就析构了
f();                                // 💥 f 里的 this 已悬空
```
