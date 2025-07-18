# C++ 内存管理

## 定位分配

普通的内存分配是在heap堆中去找寻满足条件的地址空间，定位new则是在已分配的内存区域中构造对象

对于如下所示的代码，先使用buffer变量进行空间的预申请，对于p1使用普通分配，对于p2和p3使用定位分配，在定位分配中并不会在下一次分配时自动增加地址偏移，需要我们手动进行指定。

```cpp
#include <iostream>

int main() {
    char buffer[512];
    int *p1, *p2, *p3;
    std::cout << "buffer addr " << (void*)buffer << std::endl;
    p1 = new int[10];
    std::cout << "p1 addr " << p1 << std::endl;

    p2 = new (buffer) int[10];
    std::cout << "p2 addr " << p2 << std::endl;

    p3 = new (buffer+10*4) int[10];
    std::cout << "p3 addr " << p3 << std::endl;
}
```