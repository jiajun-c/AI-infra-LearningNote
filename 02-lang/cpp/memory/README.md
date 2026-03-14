# C++ 内存管理



## malloc && free

```cpp
#include <iostream>

int main()
{
    int *arr = (int *)malloc(sizeof(int) * 10);

    free(arr);
    return 0;
}
```

## new && delete

用法为 `new type[size]`， 同时在C++11的情况下，可以在后面的括号中指定初始值，如下所示，最后输出为 `1 2 0 0 0`

```cpp
#include <iostream>

using namespace std;

int main() {
    int *arr  =new int[5]{1, 2};
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << std::endl;
    }
    delete[] arr;
}
```
### 定位分配

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

# new malloc calloc 对比

对于C++内置的类型，其实这三种函数没有本质上的区别，只是用法上不同。

malloc realloc calloc 是库函数，而new delete 是操作符

## C++ 内存分配器

C++ STL中提供了一个默认的空间配置器 `std::allocator`，我们可以用其对内存进行管理

- `alloc.allocate` 分配内存
- `alloc.deallocate` 释放内存
- `alloc.construct` 在分配的内存下构造对象
- `alloc.destroy` 析构对象

```cpp
#include <iostream>
#include <memory>

int main() {
    std::allocator<int> alloc;

    int *p = alloc.allocate(10);

    for (int i = 0; i < 10; i++) {
        alloc.construct(p + i, i);
    }
    for (int i = 0; i < 10; i++)
    {
        /* code */
        std::cout << *(p + i) << std::endl;
    }

    for (int i = 0; i < 10; i++)
    {
        /* code */
        alloc.destroy(p + i);
    }
    alloc.deallocate(p, 10);
}
```

在C++ STL 容器中默认使用 std::allocator 作为内存管理工具，但同时我们可以传入自定义的配置器

```cpp
#include <vector>
#include <iostream>

using namespace std;

int main() {
    vector<int, std::allocator<int>> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    for (int i : v) {
        cout << i << " ";
    }
    std::cout<< std::endl;
    return 0;
}
```

### C++ 自定义内存分配器

如果要自定义一个空间配置器，那么需要实现上面的 `allocator` 的四个接口，如下所示，实现了一个自定义函内存分配器，并将其用于vector的内存分配器。

```cpp
#include <iostream>
#include <vector>
using namespace std;

template <typename T>
struct MyAllocator
{
    // 必需的类型定义
    using value_type = T;

    // 允许分配器在不同类型间转换的构造函数
    template <typename U>
    MyAllocator(const MyAllocator<U>&) noexcept {}

    // // 默认构造函数
    MyAllocator() = default;

    T* allocate(size_t n) {
        std::cout << "allocate " << n << " elements" << std::endl;
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) {
        std::cout << "Deallocating " << n * sizeof(T) << " bytes." << std::endl;
        ::operator delete(p);
    }

    void construct(T* p, const T& val) {
        new(p) T(val);
    }

    void destroy(T* p) {
        p->~T();
    }
};
int main() {
    vector<int, MyAllocator<int>> v;
    v.push_back(1);
    v.push_back(2);
    for (int i = 0; i < v.size(); i++) {
        printf("%d ", v[i]);
    }
    return 0;
}
```