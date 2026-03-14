# C++ 异步操作


## 1. 异步对象生成

异步对象的产生有三种方式
- std::async
- std::packaged_task
- std::thread

`package_task` 对函数进行包装，从而返回一个 `future` 对象，其中可以存放lambda函数，bind表达式，随后 `get_future()` 获取future对象，通过get得到结果

```cpp
#include <iostream>
#include <future>
#include <functional>
#include <thread>
#include <cmath>

int f(int x, int y) {return std::pow(x, y); }

void task_lambda() {
    std::packaged_task<int(int, int)> task(
        [](int x, int y) {  return std::pow(x, y); }
    );
    std::future<int> result = task.get_future();
    task(2, 3);
    std::cout << "task_lambda: " << result.get() << std::endl;

}

void test_bind() {
    std::packaged_task<int()> task(
        std::bind(f, 2, 3)
    );
    std::future<int> result = task.get_future();
    task();
    std::cout << "test_bind: " << result.get() << std::endl;
}

void task_thread()
{
    std::packaged_task<int(int, int)> task(f);
    std::future<int> result = task.get_future();
 
    std::thread task_td(std::move(task), 2, 3);
    task_td.join();
 
    std::cout << "task_thread:\t" << result.get() << '\n';
}
 
int main() {
    task_lambda();
    test_bind();
    task_thread();
    return 0;
}
```

std::async 相比于packaged_task，std::async不需要等待任务执行完成

## 2. 异步执行


C++ future 是一个异步的库

如果要进行函数的异步执行，需要使用

