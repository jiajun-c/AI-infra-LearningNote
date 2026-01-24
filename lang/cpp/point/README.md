# 指针

## 1. 智能指针

### 1.1 shared_ptr

shared_ptr 允许对数据进行计数引用，当计数为0的时候，将会释放对应的数据，使用a.use_count()可以查看引用计数，使用a.get()可以获取裸指针。

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int>a = make_shared<int>(5);
    std::cout << "a: " << a.use_count() << "pointer " << a.get() << std::endl;
    printf("%d\n", *a);
    shared_ptr<int>b = a;
    printf("%d\n", *b);
    std::cout << "b: " << a.use_count() << "pointer " << b.get() << std::endl;
}
```


### 1.2 unique_ptr

unique_ptr 是 C++11 新增的独占型智能指针，不允许多个智能指针指向同一片内存空间，也不支持拷贝，赋值。但是支持将当前所甚指向的内存空间的
所有权交给另一个智能指针。被管理的内存空间永远只有一个智能指针指向它。

```cpp
#include <iostream>
#include <memory>

struct Task {
    int mId;
    Task(int id ) :mId(id) {
        std::cout << "Task::Constructor" << std::endl;
    }
    ~Task() {
        std::cout << "Task::Destructor " << mId <<  std::endl;
    }
};

int main()
{
    // 通过原始指针创建 unique_ptr 实例
    std::unique_ptr<Task> uniqueTaskPtr(new Task(11));
    std::unique_ptr<Task> uniqueTaskPtr1 = std::move(uniqueTaskPtr);

    printf("uniqueTaskPtr.get() = %p\n", uniqueTaskPtr.get());
    auto taskPtr =  new Task(23);

    //通过 unique_ptr 访问其成员
    int id = taskPtr->mId;
    std::cout << id << std::endl;

    return 0;
}

```

### 1.3 weak_ptr

weak_ptr 不是单独作为智能指针使用的，而是用于解决shared_ptr循环引用的问题，例如对于下面的例子

```cpp

```

## 2. 指针和数组的区别

尽管指针和数组有很多共同之点，比如他们都能通过索引来进行访问，但是有一些操作是数组做不到的，因为数组从本质上来说其实是一个不可修改的左值，它绑定了内存位置，而指针是一个变量，其只记录地址

- 数组名不可以重新赋值，如将另外一个数组名赋值给当前的数组
- 数组不可以自增或者自减
- 支持多级灵活性

同时在计算sizeof的时候，数组会返回整个的大小，但是指针不会