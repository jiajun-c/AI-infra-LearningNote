# C++ 多线程编程

## 0. 子笔记索引

- [01_basics.md](./01_basics.md) — 创建线程、joinable / join、函数传参基础
- [04_atomic_visibility.cpp](./04_atomic_visibility.cpp) — 原子可见性 + 4 种 memory_order 对比 demo

## 1. 概述

C++11 引入的标准线程库 `<thread>`，封装了平台相关的 pthread / Windows Thread，让跨平台多线程代码可移植。

核心头文件：

| 头文件 | 内容 |
|--------|------|
| `<thread>` | `std::thread` |
| `<mutex>` | `std::mutex`, `lock_guard`, `unique_lock` |
| `<condition_variable>` | `std::condition_variable` |
| `<atomic>` | 原子操作 |
| `<future>` | `std::async`, `std::promise`, `std::future` |

## 2. std::thread 基础

### 2.1 创建线程

```cpp
#include <thread>
#include <iostream>

void hello() {
    std::cout << "Hello from thread " << std::this_thread::get_id() << "\n";
}

int main() {
    std::thread t(hello);          // 启动线程
    t.join();                      // 等待线程结束（阻塞）
    // t.detach();                 // 让线程后台运行（不等待）
}
```

线程入口可以是：
- 普通函数
- lambda
- 成员函数 + 对象指针/引用
- 函数对象（重载 `operator()`）

### 2.2 join vs detach

| 方式 | 行为 | 风险 |
|------|------|------|
| `join()` | 主线程阻塞等待 | 必须调用，否则 `std::thread` 析构会 `std::terminate` |
| `detach()` | 线程独立运行 | 失去控制权，可能访问已销毁对象 |

**规则**：`std::thread` 对象销毁前，要么 `join`，要么 `detach`。

## 3. 互斥量 mutex

### 3.1 基础 mutex

```cpp
std::mutex mtx;
int counter = 0;

void worker() {
    for (int i = 0; i < 1000; ++i) {
        mtx.lock();
        ++counter;
        mtx.unlock();
    }
}
```

### 3.2 lock_guard（RAII）

```cpp
{
    std::lock_guard<std::mutex> lock(mtx);  // 构造时加锁
    ++counter;
}   // 析构时自动解锁，异常安全
```

### 3.3 unique_lock（更灵活）

```cpp
std::unique_lock<std::mutex> lock(mtx);
// 之后可以 lock.unlock() 提前解锁
// 可以 lock.lock() 再次加锁
// 配合 condition_variable 必须用 unique_lock
```

## 4. 条件变量 condition_variable

### 4.1 为什么需要 CV

mutex 解决了"互斥访问"问题，但解决不了**等待某个条件成立**。

错误做法：忙等（busy-wait）

```cpp
while (!data_ready) { /* 浪费 CPU */ }
```

正确做法：用条件变量让线程**阻塞等待**，被唤醒后再检查。

### 4.2 CV 核心 API

```cpp
std::condition_variable cv;
std::mutex mtx;
bool ready = false;

// 等待端
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock, [&]{ return ready; });   // 第二个参数是谓词，防止虚假唤醒

// 通知端
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}
cv.notify_one();    // 唤醒一个等待者
// cv.notify_all(); // 唤醒所有等待者
```

`cv.wait` 内部会：
1. 解锁 mutex（让其他线程能获取锁）
2. 阻塞当前线程
3. 被唤醒后重新加锁
4. 检查谓词，为真才返回；否则继续等

**关键**：wait 前必须持有锁，谓词检查也在锁内进行，所以等待-检查-修改是原子的。

## 5. 完整示例：生产者-消费者

见 [cv_demo.cpp](./cv_demo.cpp)

核心结构：
- 共享队列 + mutex 保护
- 两个 condition_variable 分别表示"非空"和"非满"
- 生产者：队列满则等待，否则放入并通知消费者
- 消费者：队列空则等待，否则取出并通知生产者

## 6. 常见陷阱

### 6.1 死锁

- 忘记 `join()` → 程序崩溃
- 两个 mutex 互相等待 → 锁顺序不一致
- CV wait 忘记传谓词 → 虚假唤醒 (spurious wakeup)

### 6.2 数据竞争

```cpp
int x = 0;
std::thread t([&]{ x = 1; });
std::cout << x;   // 可能读到 0，也可能读到 1，UB
t.join();
```

必须用 mutex / atomic 保护共享数据。

### 6.3 引用捕获的生命周期

```cpp
std::thread t([&]{ use(local_obj); });   // local_obj 销毁后线程还在跑？
```

线程运行期间，被引用的对象必须存活。可以用值捕获或将对象 move 到线程里。

## 7. 对比：mutex+CV vs future/promise

| 场景 | 推荐 |
|------|------|
| 一次性结果传递 | `std::future` / `std::promise` |
| 持续的生产-消费 | `std::condition_variable` |
| 简单互斥 | `std::mutex` / `std::lock_guard` |
| 无锁计数 | `std::atomic` |