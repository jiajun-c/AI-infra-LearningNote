# C++ 锁

C++ 里的「锁」有三个层次，容易混在一起讲：

```text
第 1 层: 锁本体 (mutex 家族)      —— 谁能拿、能拿几次、读写能不能并行
第 2 层: RAII 管理包装器          —— 怎么自动加锁/解锁, 怎么同时锁多把
第 3 层: 锁的实现算法             —— 自旋 / 阻塞 / 排队, 决定性能
```

日常写代码只用第 1、2 层；调性能才需要关心第 3 层。无锁算法见 [lockfree/](../lockfree/README.md)。

## 1. 锁本体：`<mutex>` 里的 mutex 家族

全部定义在 `<mutex>`，都**不可拷贝、不可移动**。

| 类型 | 标准 | 可重入 | 超时 | 读写分离 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `std::mutex` | C++11 | ✗ | ✗ | ✗ | 最常用，默认选择 |
| `std::recursive_mutex` | C++11 | ✓ | ✗ | ✗ | 同线程可重复 lock |
| `std::timed_mutex` | C++11 | ✗ | ✓ | ✗ | 支持 `try_lock_for` |
| `std::recursive_timed_mutex` | C++11 | ✓ | ✓ | ✗ | 上面两个的合体 |
| `std::shared_timed_mutex` | C++14 | ✗ | ✓ | ✓ | 读写锁 + 超时 |
| `std::shared_mutex` | C++17 | ✗ | ✗ | ✓ | 读写锁（**最常用**的读写锁） |

### 1.1 `std::mutex`

```cpp
std::mutex m;
m.lock();      // 阻塞直到拿到
bool ok = m.try_lock();   // 非阻塞, 拿不到返回 false
m.unlock();
```

`try_lock()` 是唯一"不阻塞"的入口，适合「拿不到就去做别的事」的场景。

### 1.2 `std::recursive_mutex`

同一个线程**可以重复 lock**，但必须 **unlock 相同次数**：

```cpp
std::recursive_mutex rm;
void f() { rm.lock(); /* ... */ rm.unlock(); }

void g() {
    rm.lock();     // 同一线程再次加锁, 不会死锁
    f();           // f 里又 lock 了一次 → 计数变成 2
    rm.unlock();   // 计数 1
}   // 还差一次 unlock, 这里是 bug
```

**什么时候真的需要它**：回调 / 递归调用中会重入同一把锁的代码。

**但更常见的建议是：不要用它。** 它往往说明接口设计有问题——正常的做法是把加锁的边界收窄，或者把内部实现拆成 "不加锁版本 + 加锁外壳"。因为递归锁掩盖了逻辑错误，还比普通 mutex 慢。

### 1.3 `std::timed_mutex` / `std::recursive_timed_mutex`

多两个带超时的接口：

```cpp
std::timed_mutex tm;
if (tm.try_lock_for(std::chrono::milliseconds(100))) {
    // 100ms 内拿到了
    tm.unlock();
}

if (tm.try_lock_until(deadline)) { /* ... */ }
```

用途：**避免无限期阻塞**。比如「拿不到锁就走降级路径」或者「超时就报错退出」。

### 1.4 读写锁：`std::shared_mutex`（C++17）

前提：**读多写少**。允许多个读者并行，但写者独占。

```cpp
std::shared_mutex rw;

// 读: 可以多个线程同时持有
{
    std::shared_lock<std::shared_mutex> lk(rw);   // lock_shared()
    read_data();
}

// 写: 独占
{
    std::unique_lock<std::shared_mutex> lk(rw);   // lock()
    write_data();
}
```

注意 api 命名不太对称：

| 语义 | 直接调用 | 对应的 RAII |
| --- | --- | --- |
| 写锁（独占） | `lock()` / `unlock()` | `std::unique_lock` |
| 读锁（共享） | `lock_shared()` / `unlock_shared()` | `std::shared_lock` |

**什么时候用**：读远多于写（比如配置表、路由表、缓存索引）。如果读写差不多，`std::mutex` 反而更快——读写锁自己也有开销，而且**写者可能饿死**（读者源源不断）。

C++14 的 `std::shared_timed_mutex` 多支持超时；C++17 的 `std::shared_mutex` 没有超时但通常更快。

## 2. RAII 包装器：怎么用锁

**永远不要手写 `lock()` / `unlock()`**——异常路径、提前 return 都会漏掉 unlock。用 RAII 包装器。

| 包装器 | 标准 | 锁几把 | 能手动解锁 | 能延迟加锁 | 能移动 | 典型用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `std::lock_guard` | C++11 | 1 | ✗ | ✗ | ✗ | 简单临界区，**默认选择** |
| `std::unique_lock` | C++11 | 1 | ✓ | ✓ | ✓ | 配合 condvar、需要中途解锁 |
| `std::scoped_lock` | C++17 | N | ✗ | ✗ | ✗ | **多把锁**，防死锁 |
| `std::shared_lock` | C++14 | 1 | ✓ | ✓ | ✓ | 配合 `shared_mutex` 读锁 |

### 2.1 `std::lock_guard` — 默认选择

```cpp
{
    std::lock_guard<std::mutex> lk(m);
    ++counter;
}   // 析构自动 unlock
```

开销最小（编译器通常能完全优化掉）。**能用它就用它。**

### 2.2 `std::unique_lock` — 灵活但有代价

```cpp
// 1) 延迟加锁
std::unique_lock<std::mutex> lk(m, std::defer_lock);
do_something_else();
lk.lock();

// 2) 提前解锁
lk.unlock();
do_slow_thing();     // 锁外做

// 3) 配合 condition_variable (必须用它)
std::unique_lock<std::mutex> lk(m);
cv.wait(lk, [] { return ready; });
```

因为它要支持"当前是否持有锁"的状态，比 `lock_guard` 多存一个 bool、多一层分支，**不要无脑用**。

`std::defer_lock` / `std::try_to_lock` / `std::adopt_lock` 三个 tag：

| tag | 含义 |
| --- | --- |
| `std::defer_lock` | 先不加锁，之后手动 lock |
| `std::try_to_lock` | 构造时 try_lock，不阻塞 |
| `std::adopt_lock` | 假设锁已经被 lock 了，只负责解锁 |

### 2.3 `std::scoped_lock` — 多把锁的正确姿势（C++17）

经典死锁场景：

```cpp
// 线程 A                     // 线程 B
std::lock_guard g1(m1);        std::lock_guard g1(m2);
std::lock_guard g2(m2);        std::lock_guard g2(m1);   // ← 顺序相反
// 可能死锁
```

`std::scoped_lock` 内部用 `std::lock()` 的**死锁避免算法**（try-and-back-off）原子地锁住所有 mutex：

```cpp
std::scoped_lock lk(m1, m2);   // 不会死锁, 无论 m1/m2 的加锁顺序
```

它是 `lock_guard` 的可变参数版，**N 把锁就用它**。

配套的还有：

```cpp
int idx = std::try_lock(m1, m2, m3);   // 全部拿到返回 -1, 否则返回第一个失败的索引
std::lock(m1, m2);                     // 阻塞版, 但需要自己 unlock
```

### 2.4 `std::shared_lock` — 读锁的 RAII

```cpp
std::shared_mutex rw;
{
    std::shared_lock<std::shared_mutex> lk(rw);   // 共享/读锁
    // 多个线程可以同时进这里
}
{
    std::unique_lock<std::shared_mutex> lk(rw);   // 独占/写锁
    // 只有一个线程能进
}
```

## 3. 锁的实现算法

同一个 `std::mutex` 接口，底层可以完全不同的实现。这是"锁"话题里性能差异最大的部分。

### 3.1 自旋锁（spinlock）

拿不到锁就**忙等**（反复检查），不进内核。

```cpp
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // 空转
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }
};
```

| 优点 | 缺点 |
| --- | --- |
| 无系统调用，短临界区极快 | 空转烧 CPU |
| 无上下文切换 | 所有核抢同一 cacheline → **cache line bouncing** |
| 适合锁持有时间 < 线程切换开销 | 持有久时性能灾难 |

**适用**：临界区只有几条指令（比如改一个计数器）。

CAS 版、Ticket lock、MCS lock 的实现见 [lockfree/README.md](../lockfree/README.md)。

### 3.2 阻塞锁

拿不到锁就**睡眠**，让出 CPU，由内核唤醒。

- 无空转浪费，适合长临界区
- 但有两次上下文切换开销（睡 + 醒），通常几微秒

Linux 上 `std::mutex` 底层是 **futex**（fast userspace mutex）：无竞争时纯用户态原子操作，有竞争才陷入内核。

### 3.3 混合锁（自适应锁）

**先自旋一小会儿，还拿不到再睡眠**。这是 glibc `std::mutex` 的默认策略，兼顾两者：

```text
try_lock 成功          → 直接进 (最快路径, 无系统调用)
try_lock 失败, 自旋几次 → 短暂等待, 避免立即睡眠
还失败                 → futex 睡眠, 让出 CPU
```

现代 libstdc++ / libc++ 的 `std::mutex` 都是这种自适应实现。

### 3.4 各种锁算法对比

| 算法 | 公平性 | 等待方式 | cache 友好 | 备注 |
| --- | --- | --- | --- | --- |
| TAS / CAS 自旋 | 不保证 | 自旋同一变量 | ✗ | 简单，易饿死 |
| Ticket lock | FIFO 公平 | 自旋各自 ticket | ✗ | 公平但仍有 bouncing |
| MCS lock | FIFO 公平 | 自旋**本地**变量 | ✓ | 链表排队，工业级 |
| CLH lock | FIFO 公平 | 自旋前驱节点 | ✓ | 类似 MCS |
| futex / 阻塞锁 | 不保证 | 内核睡眠 | — | 长临界区 |

MCS 的关键改进：每个等待者**自旋在自己的本地节点上**，不碰全局变量，避免所有核抢同一条 cacheline。详见 [lockfree/](../lockfree/README.md)。

## 4. 不用锁的同步原语

有些同步需求根本不需要 mutex：

| 原语 | 头文件 | 标准 | 用途 |
| --- | --- | --- | --- |
| `std::atomic<T>` | `<atomic>` | C++11 | 单个变量的原子读写 |
| `std::atomic_ref<T>` | `<atomic>` | C++20 | 对已有变量做原子访问 |
| `std::once_flag` / `std::call_once` | `<mutex>` | C++11 | 只执行一次（单例初始化） |
| `std::counting_semaphore` | `<semaphore>` | C++20 | 计数信号量，限流 |
| `std::binary_semaphore` | `<semaphore>` | C++20 | 二值信号量 = `counting_semaphore<1>` |
| `std::latch` | `<latch>` | C++20 | 一次性屏障（等 N 个线程到齐） |
| `std::barrier` | `<barrier>` | C++20 | 可复用的屏障 |

**`std::call_once`** 是最值得记住的一个——它比"用 mutex 保护一个 bool"更快也更安全：

```cpp
std::once_flag flag;
void init() {
    std::call_once(flag, [] {
        // 无论多少线程调用, 这里只执行一次
    });
}
```

**C++20 的 `atomic::wait/notify`** 也很实用，可以用它手写更轻量的同步：

```cpp
std::atomic<int> state{0};
// 等待方
state.wait(0);          // 阻塞直到 state != 0
// 通知方
state.store(1);
state.notify_one();
```

底层通常直接映射到 futex，比 `mutex + condvar` 轻。

## 5. 选型速查

```text
需要保护共享数据?
├─ 只是一个计数器 / 标志位
│    → std::atomic (不用锁)
│
├─ 只初始化一次
│    → std::call_once
│
├─ 读远多于写
│    → std::shared_mutex + shared_lock / unique_lock
│
├─ 要同时锁多把
│    → std::scoped_lock
│
├─ 临界区极短 (几条指令) 且多核竞争激烈
│    → 自旋锁 (自己实现, 或看 lockfree/)
│
├─ 普通临界区 (默认情况)
│    → std::mutex + std::lock_guard
│
├─ 需要在临界区中间解锁 / 配合 condvar
│    → std::unique_lock
│
└─ 回调/递归里会重入同一把锁
     → 先想想能不能重构; 实在不行才 std::recursive_mutex
```

## 6. 常见坑

| 坑 | 后果 | 正确做法 |
| --- | --- | --- |
| 手写 `lock()` / `unlock()` | 异常路径漏 unlock → 死锁 | 用 RAII 包装器 |
| 两把锁顺序不一致 | 死锁 | `std::scoped_lock` 或固定顺序 |
| 在持锁时做 IO / sleep | 吞吐塌方 | 缩小临界区，慢活挪到锁外 |
| 持锁时调用未知回调 | 回调里再加锁 → 死锁 | 不在锁内调外部代码 |
| `shared_mutex` 用在写多的场景 | 比 `mutex` 还慢 | 读写比不悬殊就用 `mutex` |
| `recursive_mutex` 当万能药 | 掩盖设计问题，更慢 | 收窄锁边界，拆内外层 |
| `unique_lock` 无脑用 | 比 `lock_guard` 多一层开销 | 不需要灵活度就用 `lock_guard` |
| 锁的粒度太粗 | 并发度上不去 | 分段锁 / 每对象一把锁 |

**关于锁粒度**：从粗到细的演进一般是

```text
全局一把大锁  →  分段锁 (sharded lock)  →  每对象一把锁  →  无锁
  简单但串行      按 key hash 分桶         并发度最高       最难写
```

## 7. 相关笔记

- [C++ 多线程基础](../../02-lang/cpp/thread/README.md) — thread / mutex 入门 / condition_variable
- [Lock-free 算法](../lockfree/README.md) — CAS 自旋锁、Ticket lock、MCS lock 实现与 benchmark
- [内存序](../../02-lang/cpp/memory/memoryOrder.md) — `acquire` / `release` / `relaxed` 语义
