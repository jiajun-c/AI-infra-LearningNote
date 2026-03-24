#  Vector

## 1. vector类原理

vector类是动态数组，随着数据的加入，他的内部机制将会自动扩充来容纳新元素。

vector的内存管理由M_impl中的M_start, M_finish, M_end_of_storage三个指针来管理。所有关于地址，容量大小等操作都要用到这三个指针。

- M_start: 指向迭代器位置的指针
- M_finish: 指向迭代器结束位置的指针
- M_end_of_storage: 存储空间结束位置的指针

## 2. vector类的拷贝

### 2.1 深拷贝

vector作为一个普通类，其分为深拷贝和浅拷贝，如下所示，打印出来的地址其实是不同的。

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;
    v.push_back(1);

    std::vector<int>v1 = v;
    printf("v addr %0x\n", v.data());
    printf("v1 addr %0x\n", v1.data());

    std::vector<int>v2 = {1, 2, 3, 4, 5};
    std::vector<int>v3(v1.begin(), v1.begin() + 3);
    printf("v2 addr %x\n", v2.data());
    printf("v3 addr %x\n", v3.data());

}
// v addr d7cc2eb0
// v1 addr d7cc2ed0
// v2 addr d7cc3300
// v3 addr d7cc3320
```

这些地址都是不一样的

### 2.2 浅拷贝

使用浅拷贝的话，如std::span去创建一个原vector的实图，那么元素的起始地址都是一样的，std::span是C++20才引入的特性，也可以使用`std::vector<int>&v1`去做一个引用

```cpp
#include <iostream>
#include <vector>
#include <span>
#include <cstdio> // 修复 1：必须包含此头文件

using namespace std;

int main() {
    vector<int> v1 = {1, 2, 3, 4};
    std::span<int> view_v1(v1);
    // 修复 2：打印指针地址使用 %p，避免类型截断和编译器报错
    printf("v1: %p\n", (void*)v1.data());
    printf("v2: %p\n", (void*)view_v1.data());
    
    return 0;
}
```