# 多线程 (C++ Concurrency)

C++11 之后有一套完整的多线程原语，定义在 `<thread>`、`<mutex>`、`<condition_variable>`、`<atomic>`、`<future>` 五个 header 里。这一节按 **由浅入深** 的顺序，把这些原语过一遍，每一节都有可独立编译运行的最小例子。

## 学习路径

| # | 主题 | 关键 API | 学完能做什么 |
| --- | --- | --- | --- |
| 01 | 线程基础 | `std::thread`、`join`、`detach` | 启动并等待一个后台线程 |
| 02 | 互斥 | `std::mutex`、`lock_guard`、`scoped_lock` | 保护共享数据，避免 data race |
| 03 | 条件变量 | `std::condition_variable` | 「等到某事件发生」式的同步 |
| 04 | 原子操作 | `std::atomic<T>`、`fetch_add` | 不加锁的计数器、标志位 |
| 05 | 一次性同步 | `std::future`、`std::promise`、`std::async` | 把"返回值从子线程带回主线程"语义化 |
| 06 | 线程池 | 综合 | 一个能复用的 worker pool |
| 07 | 陷阱 | — | data race / deadlock / 假唤醒 / lost wakeup |
| 终 | MPMC 队列 | 综合 | 多个生产者 + 多个消费者 + 安全 shutdown |

## 编译运行所有示例

```bash
cd 07-system/cpu/multiThread
for f in 0*.cpp; do
  echo "=== $f ==="
  g++ -std=c++17 -O2 -Wall -pthread "$f" -o "/tmp/${f%.cpp}"
  "/tmp/${f%.cpp}"
  echo
done
```

## 推荐阅读顺序

1. 把 01 到 05 **按顺序**过一遍，每节把代码读懂、改几处自己跑一下。
2. 07 是必读，里面列的几个 bug 几乎是所有人都踩过的。
3. 06 和终极的 MPMC 队列是"综合运用"，建议学完前 5 节再回头看。
4. 看到 `std::memory_order_*` 之类的字眼觉得陌生，先放一放，那是更深的内存模型话题，本节只用到最朴素的默认 `seq_cst`。

## 各节核心要点速查

### 01 — 线程基础

- `std::thread t([]{ ... })`：启动一个 OS 线程，**几乎立即开始执行**。
- `t.join()`：阻塞等它结束。**不 join 也不 detach，析构时会 std::terminate**。
- `t.detach()`：让线程在后台独立运行，**之后不能再 join**。
- 线程函数的所有参数都**按值拷贝**进线程（除非 `std::ref`）。捕获要传引用时用 lambda。
- `std::this_thread::get_id()` / `sleep_for()` / `yield()` 是最常用的几个工具函数。

### 02 — 互斥

- 共享数据被多线程读写时，必须有同步，否则就是 data race（未定义行为）。
- `std::mutex m; m.lock(); ... m.unlock();`：**永远不要手动 lock/unlock**，异常路径会泄漏。
- `std::lock_guard<std::mutex> g(m);`：RAII 包装，析构自动 unlock。**这是默认选择**。
- `std::scoped_lock(m1, m2, ...)`：同时锁多把 mutex，**自动按避免死锁的顺序加锁**（C++17）。
- `std::unique_lock`：比 `lock_guard` 更灵活，可以提前 unlock / 配合 condvar。
- 死锁四条件：互斥、占有并等待、不可剥夺、循环等待。**永远按固定顺序加锁**。

### 03 — 条件变量

- `condvar` 必须配合 `std::unique_lock<std::mutex>` 用。
- **永远在循环里 wait**，因为存在「假唤醒」(spurious wakeup) 和「lost wakeup」。
- `wait(lk, predicate)` 是「while (!predicate) wait(lk)」的语法糖，更安全。
- `notify_one` 唤醒**一个**等待者，`notify_all` 唤醒**全部**。默认用 `notify_one`，避免惊群。
- 修改 predicate 关联的状态时必须持锁，notify 之前**先 unlock 再 notify**（减少 context switch）。

### 04 — 原子操作

- `std::atomic<T>` 把 T 的单次读写变成不可分割的硬件原子操作。
- `fetch_add / fetch_sub / compare_exchange_weak/strong` 是常用原语。
- 默认 `seq_cst`（顺序一致）最安全、最慢；`acquire/release` 比 `seq_cst` 快、约束弱；`relaxed` 最快、最弱。
- 简单计数器、标志位用 `std::atomic<bool>` / `std::atomic<int>` 是 lock-free 的好场景。
- **复合操作必须用 `compare_exchange`**，不能"读 + 写"分两步。

### 05 — 一次性同步

- `std::promise<T>`：写端。`p.set_value(x)` 把值 x 传出去。
- `std::future<T>`：读端。`f.get()` 阻塞等 promise 给值；`f.wait_for(t)` 不阻塞等超时。
- `std::async(std::launch::async, []{ ... })` 返回一个 future，**最常用**，比手动 promise + future 短很多。
- `std::shared_future<T>`：一份 future 可以被多个线程 `.get()`，但 `std::future` 只能 get 一次。

### 06 — 线程池

- 一个固定数量的 worker 线程，从任务队列里反复取任务执行。
- 内部就是 mutex + condvar + queue + 一组 thread 的组合。
- shutdown 语义：把队列排干后退出。

### 07 — 陷阱

| 陷阱 | 现象 | 解决 |
| --- | --- | --- |
| Data race | 结果不确定 | mutex / atomic |
| 死锁 | 所有线程卡住 | 固定加锁顺序 / `std::scoped_lock` |
| 假唤醒 | wait 莫名返回 | `while (!pred) wait(lk)` |
| Lost wakeup | notify 早于 wait，信号丢失 | wait 用 predicate 版本；notify 前持锁 |
| 忘记 join | std::terminate | RAII / join 助手 |
| 在锁里 sleep / IO | 吞吐塌方 | 缩小临界区 |
| `std::vector` push_back 多线程 | UB（realloc） | 预 reserve / mutex / tbb |

## 终极示例

- [`mpmc_queue.hpp`](./mpmc_queue.hpp)：把 02、03 全部用上的 MPMC 队列实现。
- [`gpu_pipeline_demo.cpp`](./gpu_pipeline_demo.cpp)：在 MPMC 上跑一个模拟 GPU 流水线。

读完前面 7 节再回来看这两个文件，会觉得"原来就这几招"。

## 推荐外部资料

- **《C++ Concurrency in Action》** (Anthony Williams)：公认最系统的 C++ 并发书，第二版覆盖到 C++17。
- **cppreference.com** 的 [thread](https://en.cppreference.com/w/cpp/thread) / [atomic](https://en.cppreference.com/w/cpp/atomic) 页面：每个 API 都有完整示例。
- **Herb Sutter 的 "Effective Concurrency"** 系列：偏高层设计，跨语言也通用。
