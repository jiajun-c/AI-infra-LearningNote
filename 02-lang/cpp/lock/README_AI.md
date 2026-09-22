# C++ lock 机制

C++ 的「锁」分三层，混在一起讲最容易乱：

```text
第 1 层: 锁本体 (mutex 家族)   —— 谁能拿、能拿几次、读写能不能并行
第 2 层: RAII 管理包装器       —— 怎么自动加锁/解锁, 怎么同时锁多把
第 3 层: 锁的实现算法          —— 自旋 / 阻塞 / 排队, 决定性能
```

本文讲**第 1 层**：`<mutex>` 和 `<shared_mutex>` 里的 6 个锁类型。
第 2 层（`lock_guard` / `unique_lock` / `scoped_lock` / `shared_lock`）见 [锁总览](../../../10-dist/lock/README.md)。
第 3 层（自旋锁 / Ticket / MCS）见 [lockfree/](../../../10-dist/lockfree/README.md)。

配套代码：[mutex.cpp](./mutex.cpp)，本文所有数字都是它在本机跑出来的。

## 1. 六个类型总览

全部定义在 `<mutex>`（`shared_mutex` 在 `<shared_mutex>`），**都不可拷贝、不可移动**。

| 类型 | 标准 | 头文件 | 可重入 | 超时 | 读写分离 |
| --- | --- | --- | --- | --- | --- |
| `std::mutex` | C++11 | `<mutex>` | ✗ | ✗ | ✗ |
| `std::recursive_mutex` | C++11 | `<mutex>` | ✓ | ✗ | ✗ |
| `std::timed_mutex` | C++11 | `<mutex>` | ✗ | ✓ | ✗ |
| `std::recursive_timed_mutex` | C++11 | `<mutex>` | ✓ | ✓ | ✗ |
| `std::shared_timed_mutex` | C++14 | `<shared_mutex>` | ✗ | ✓ | ✓ |
| `std::shared_mutex` | C++17 | `<shared_mutex>` | ✗ | ✗ | ✓ |

三个正交的特征，组合出这 6 个类型：

```text
        可重入?     超时?      读写分离?
mutex      ✗          ✗           ✗      ← 默认
recursive  ✓          ✗           ✗
timed      ✗          ✓           ✗
rec_timed  ✓          ✓           ✗
shared_t   ✗          ✓           ✓
shared     ✗          ✗           ✓      ← C++17, 最常用的读写锁
```

## 2. 逐个详解

### 2.1 `std::mutex` — 默认选择

完整的接口只有三个方法：

```cpp
std::mutex m;

m.lock();                    // 阻塞直到拿到锁
bool ok = m.try_lock();      // 非阻塞, 拿不到立刻返回 false
m.unlock();                  // 解锁
```

语义要点：

- **构造函数是 `constexpr`**，所以可以安全地做全局对象，不会有静态初始化顺序问题（SIOF）。
- `lock()` 阻塞，不计超时。
- `try_lock()` **允许虚假失败**（spurious failure）：即使没有任何其他线程持锁，它也可能返回 `false`。所以**不能**用它来判断"有没有竞争"。
- 标准**不保证公平**，高竞争下某个线程可能饿死。
- 同一线程**重复 `lock()` 是 UB**（典型表现是死锁）——见 [§5](#5-语义细节与-ub-清单)。

**这是默认选择。** 除非你有明确的理由（需要重入 / 超时 / 读写分离），否则就用它。

```cpp
std::mutex m;
{
    std::lock_guard<std::mutex> lk(m);   // 永远用 RAII, 不要手写 lock/unlock
    ++counter;
}
```

### 2.2 `std::recursive_mutex` — 同线程可重入

接口和 `std::mutex` 完全一样，区别只在语义：**同一个线程可以重复加锁**，内部维护 `owner tid` + 递归计数。

```cpp
std::recursive_mutex rm;

void f() { std::lock_guard<std::recursive_mutex> lk(rm); /* ... */ }
void g() {
    std::lock_guard<std::recursive_mutex> lk(rm);   // 计数 1
    f();                                            // 计数 2, 不死锁
}                                                   // 递减到 0 才真正释放
```

- 必须 **unlock 相同次数**才真正释放，靠 RAII 才能保证配平。
- 最大递归深度**标准未指定**，超过会抛 `std::system_error`。
- 比 `std::mutex` 慢（实测有竞争时约 1.3 倍，见 [§4](#4-实测数据)）。

**什么时候真的需要**：回调 / 递归调用路径中会重入同一把锁，且短期内无法重构。

**但更常见的建议是：不要用它。** 它几乎总意味着接口设计有问题——调用方无法知道"这个函数内部会不会重入"。正确做法通常是二选一：

```cpp
// 方案 A: 拆成「不加锁实现 + 加锁外壳」
class Counter {
    int v_ = 0;                     // 内部实现, 假定调用方已持锁
    int get_unlocked() const { return v_; }
public:
    int get() { std::lock_guard<std::mutex> lk(m_); return get_unlocked(); }
private:
    std::mutex m_;
};

// 方案 B: 收窄锁边界, 让重入路径不再需要同一把锁
```

### 2.3 `std::timed_mutex` / `std::recursive_timed_mutex`

在 `std::mutex` 三个方法之上，多两个**带超时**的接口：

```cpp
std::timed_mutex tm;

tm.try_lock_for(std::chrono::milliseconds(100));         // 相对超时
tm.try_lock_until(std::chrono::steady_clock::now() + d); // 绝对时间点
```

两者的区别只在"超时怎么表达"：`for` 用时长，`until` 用时间点（更适合循环里算 deadline）。

**用途是"避免无限期阻塞"**：

```cpp
if (tm.try_lock_for(std::chrono::milliseconds(50))) {
    // 拿到了, 正常路径
    do_work();
    tm.unlock();
} else {
    // 50ms 还没拿到 —— 说明锁竞争严重或者有死锁
    // 走降级路径: 用旧数据 / 报错 / 打日志
    log_contention_warning();
}
```

`std::recursive_timed_mutex` 就是这两个的并集（可重入 + 超时）。

注意：**超时返回 `false` 不代表锁一定被别人拿着**——和 `try_lock` 一样允许虚假失败，也可能是调度延迟。

### 2.4 `std::shared_mutex` — 读写锁（C++17）

前提：**读多写少**。允许多个读者并行，写者独占。

API 分成两组，命名不太对称，这是最容易记错的地方：

| 语义 | 直接调用 | 对应的 RAII |
| --- | --- | --- |
| 写锁（独占 / exclusive） | `lock()` / `try_lock()` / `unlock()` | `std::unique_lock` |
| 读锁（共享 / shared） | `lock_shared()` / `try_lock_shared()` / `unlock_shared()` | `std::shared_lock` |

```cpp
std::shared_mutex rw;

// 读: 多个线程可以同时进来
{
    std::shared_lock<std::shared_mutex> lk(rw);
    read_data();
}

// 写: 独占, 与其他读写都互斥
{
    std::unique_lock<std::shared_mutex> lk(rw);
    write_data();
}
```

`std::shared_lock` 只能配 `shared_mutex` 用（它调的是 `lock_shared()`），`std::unique_lock` 配普通 mutex 或 `shared_mutex` 的写锁都行。

三个必须知道的限制：

1. **不可重入**。读锁和写锁都不能递归——同一线程拿了读锁再拿写锁会死锁。
2. **不支持锁升级**。不能从"持有读锁"直接变成"持有写锁"，必须先 `unlock()` 再 `lock()`。所以下面的写法是**死锁**：

   ```cpp
   // 错误! 读锁和写锁在同一线程上互相等
   std::shared_lock<std::shared_mutex> rlk(rw);
   if (need_fix()) {
       std::unique_lock<std::shared_mutex> wlk(rw);   // ← 死锁
   }
   ```

3. **写者可能饿死**。读者源源不断时，写者可能一直拿不到锁。标准不保证公平性。

**什么时候用**：读远多于写（配置表、路由表、缓存索引）。如果读写比不悬殊，**用 `std::mutex` 反而更快**——读写锁自己也有开销，实测无竞争时它的写锁就要 21 ns，是普通 mutex 的 5 倍。见 [§4](#4-实测数据)。

`std::shared_timed_mutex`（C++14）是它的前身，多支持四个超时接口：

```cpp
try_lock_for() / try_lock_until()                  // 写锁超时
try_lock_shared_for() / try_lock_shared_until()    // 读锁超时
```

C++17 的 `shared_mutex` 砍掉了超时，但实现上可以更轻。**新代码优先用 `std::shared_mutex`。**

## 3. 命名要求（这些类型要满足什么概念）

标准用「命名要求」描述锁的接口契约。看懂这张表，就明白 6 个类型为什么这么分层：

| 概念 | 要求的方法 | 谁满足 |
| --- | --- | --- |
| **Lockable** | `lock` `try_lock` `unlock` | 全部 6 个 |
| **TimedLockable** | Lockable + `try_lock_for` `try_lock_until` | `timed_mutex`、`recursive_timed_mutex`、`shared_timed_mutex` |
| **SharedLockable** (`SharedMutex`) | Lockable + `lock_shared` `try_lock_shared` `unlock_shared` | `shared_mutex`、`shared_timed_mutex` |
| **SharedTimedLockable** | 上面两者的并集 | `shared_timed_mutex` |

**这个概念体系的实际价值**：`std::lock_guard<M>`、`std::unique_lock<M>`、`std::lock(a, b, ...)` 都是模板函数，它们的模版参数约束就是这些概念。所以**你自己写的类只要实现了 `lock`/`unlock`，就能直接塞进 `std::lock_guard`**：

```cpp
class MyLock {
public:
    void lock()   { /* ... */ }
    void unlock() { /* ... */ }
};

MyLock ml;
std::lock_guard<MyLock> lk(ml);   // 可以用, 只要满足 Lockable
```

这也是为什么日志锁、文件锁、自定义自旋锁都能无缝接入标准库的 RAII 体系。

## 4. 实测数据

来自 [mutex.cpp](./mutex.cpp)，本机 x86-64 / libstdc++ / -O2。

### 4.1 对象本身的大小

| 类型 | sizeof |
| --- | --- |
| `std::mutex` | 40 |
| `std::recursive_mutex` | 40 |
| `std::timed_mutex` | 40 |
| `std::recursive_timed_mutex` | 40 |
| `std::shared_mutex` | 56 |
| `std::shared_timed_mutex` | 56 |

40 = glibc 里 `pthread_mutex_t` 的大小；56 = `pthread_rwlock_t`。读写锁大 16 字节，因为它要分别维护读者计数和写者状态。

对比 `sizeof(std::atomic<int>) = 4` —— **锁的元数据比数据本身贵 10 倍**，这是"能用 atomic 就别用锁"的一个直观理由。而且这是实现定义的，MSVC 上 `std::mutex` 明显更小（它用 `SRWLOCK`）。

### 4.2 无竞争开销（单线程 1e7 次 lock+unlock）

| 类型 | ns/op |
| --- | --- |
| `std::mutex` | 4.4 |
| `std::recursive_mutex` | 4.1 |
| `std::timed_mutex` | 3.2 |
| `std::shared_mutex`（写锁） | **21.2** |

前三个都在 3–4 ns，因为无竞争时**都是纯用户态原子操作，不陷内核**。这几个数字的差异在测量噪声范围内。

`shared_mutex` 的写锁要 21 ns，**是普通 mutex 的 5 倍**——因为拿写锁时必须检查是否有活跃读者，逻辑本身就重。这就是"读写比不悬殊就别用读写锁"的量化依据。

### 4.3 有竞争开销（2 线程抢同一把锁）

跑 5 次取中位数（括号是波动范围）：

| 类型 | ns/op（中位数） | 波动范围 | 相对 `mutex` |
| --- | --- | --- | --- |
| `std::mutex` | 41.5 | 37.9 – 49.5 | 1.0x |
| `std::recursive_mutex` | 53.5 | 45.5 – 57.1 | 1.3x |
| `std::shared_mutex`（写锁） | 101.8 | 80.4 – 113.6 | 2.5x |

有竞争时从 4 ns 涨到约 41 ns（**约 10 倍**），因为要陷入 futex 睡眠/唤醒，涉及两次上下文切换。递归锁额外贵在比较 owner tid。

**注意噪声**：有竞争的微基准波动很大（±20%），单次运行三个类型的相对排序都可能抖动。**要下结论必须多跑几次取中位数**——上面这张表是 5 次的统计，不是一次运行的结果。

### 4.4 读写锁真的能并行读吗（4 线程 × 200k）

同样 5 次取中位数：

| 场景 | ns/op（中位数） | 波动范围 |
| --- | --- | --- |
| 4 个读者（`shared_lock`） | 84.8 | 75.7 – 92.2 |
| 4 个写者（`unique_lock`） | 146.3 | 133.4 – 154.3 |

读者只比写者快 **1.7 倍，而不是 4 倍**。原因是这个 benchmark 的临界区只有一条加法——**读者之间仍然要争抢同一个读者计数的 cacheline**，并行收益被这个开销吃掉了。

**结论**：读写锁的并行收益取决于**临界区有多长**。临界区里做的事越多（比如遍历一个几百项的 map），读并行的收益才越接近线性。临界区只有几条指令时，`std::mutex` 更划算。

## 5. 语义细节与 UB 清单

这些是标准里明确规定的，踩了不会报错，只会莫名其妙地挂：

| # | 写法 | 后果 |
| --- | --- | --- |
| a | 忘记 `unlock` / 中途 `return` | 死锁（用 RAII 解决） |
| b | 非递归 mutex 同线程重复 `lock()` | **UB**，典型表现是死锁 |
| c | `unlock()` 一把本线程没持有的锁 | **UB** |
| d | 析构时 mutex 仍被锁住 | **UB** |
| e | 把 `try_lock()` 返回 `false` 当"有人在用" | 逻辑错误（允许虚假失败） |
| f | 假设锁是公平的 | 可能饿死某个线程 |
| g | 在持有锁时销毁锁对象 | **UB** |

补充说明：

- **(b)** 对 `std::mutex` 是 UB，对 `std::recursive_mutex` 是定义良好的（这就是它存在的意义）。demo 里用 `try_lock` 安全地演示了这一点。
- **(e)** 是最隐蔽的。`try_lock` 的实现可能是"CAS 失败就返回"，硬件竞争、cacheline 迁移都会让它失败，即使逻辑上没冲突。
- **(f)** 标准只要求"最终能拿到"，不要求 FIFO。Ticket lock / MCS lock 才是公平的，见 [lockfree/](../../../10-dist/lockfree/README.md)。

另外两个容易忽略的：

- **锁不能拷贝**。它是独占资源的所有权凭证，拷贝语义没有意义。这也意味着**不能放进需要拷贝的容器**（`std::vector` 扩容就需要），只能 `std::deque` / `std::list` / `unique_ptr` 持有。
- **`std::mutex` 的构造函数是 `constexpr`**，所以全局 mutex 不会遇到静态初始化顺序问题；但 `shared_mutex` 不是，全局 `shared_mutex` 要小心。

## 6. 底层实现

标准只规定接口，实现完全自由。

### Linux / libstdc++

```text
std::mutex          →  pthread_mutex_t   (40 字节)
std::recursive_mutex →  pthread_mutex_t + PTHREAD_MUTEX_RECURSIVE
std::shared_mutex   →  pthread_rwlock_t  (56 字节)
```

glibc 的 `pthread_mutex_t` 是**自适应混合锁**，基于 **futex**（fast userspace mutex）：

```text
lock():
  1. 先试一次用户态原子 CAS     → 成功就直接进 (无系统调用, ~4ns)
  2. 失败则自旋几次             → 短暂等待, 避免立刻睡眠
  3. 还失败才 futex 陷入内核睡眠 → 让出 CPU (涉及上下文切换, ~50ns+)

unlock():
  1. 用户态原子释放
  2. 如果有等待者, futex_wake 唤醒一个
```

这就是为什么"无竞争 4 ns"和"有竞争 49 ns"差 10 倍——**慢的不是锁本身，是内核态往返**。

读写锁 `pthread_rwlock_t` 额外维护读者计数，所以写锁要检查计数器状态，无竞争也慢（21 ns）。

### 其他平台的差异

- **MSVC**：`std::mutex` 用 `SRWLOCK`，对象小得多（一个指针）。所以**不要跨平台假设 `sizeof`**。
- **macOS / libc++**：`std::mutex` 在较新版本是自旋 + `psynch` 的混合实现。

## 7. 选型

```text
需要保护共享数据?
│
├─ 只是一个计数器 / 标志位
│    → std::atomic, 不用锁  (sizeof 4 vs 40)
│
├─ 读远多于写, 且临界区不短
│    → std::shared_mutex + shared_lock / unique_lock
│    → 临界区很短的话, 还是 std::mutex 更快
│
├─ 需要"拿不到就走别的路" / 怕死锁
│    → std::timed_mutex + try_lock_for
│
├─ 回调 / 递归路径里会重入同一把锁
│    → 先想能不能重构 (拆成不加锁实现 + 加锁外壳)
│    → 实在不行才 std::recursive_mutex
│
└─ 普通临界区 (90% 的情况)
     → std::mutex + std::lock_guard
```

一句话经验：

```text
默认用 std::mutex。
另外 5 个类型都是"有明确理由才选", 而不是"看起来更强所以选"。
```

## 8. 相关笔记

- [C++ 锁总览](../../../10-dist/lock/README.md) — 三层结构 + 选型速查
- [Lock-free 算法](../../../10-dist/lockfree/README.md) — 自旋锁 / Ticket / MCS 的实现与 benchmark
- [C++ 多线程基础](../thread/README.md) — `std::thread` / `condition_variable` 入门
- [内存序](../memory/memoryOrder.md) — `acquire` / `release` / `relaxed` 语义
