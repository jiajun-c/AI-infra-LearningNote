# C++ 内存序

## 1. relax

`relax`不保证操作都被原子地执行，但是不保证操作的顺序，适用于计数器等场景

```cpp
#include <atomic>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>

std::atomic<long> g_counter{0};

void worker(int n) {
    for (int i = 0; i < n; i++) {
        g_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

int main() {
    std::vector<std::thread>ts;
    for (int i = 0; i < 8; i++) {
        ts.emplace_back(worker, 1000);
    }

    for (auto &t: ts) {
        t.join();
    }

    printf("%d\n", g_counter.load(std::memory_order_relaxed));
}
```

## 2. release/acquire

release保证上面访存在这个时候都是可见的，不会被重排到下面。

acquire保证下面的访存不会被重排到上面。

release和acquire的组合可以用于生产者和消费者的模式

```cpp
#include <atomic>
#include <thread>
#include <cassert>
#include <string>

struct Payload
{
    int a;
    std::string s;
};

Payload g; 

std::atomic<bool>g_ready(false);

void producer() {
    g.a = 42;
    g.s = "hello";
    g_ready.store(true, std::memory_order_release);
}

void consumer() {
    while (!g_ready.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
    assert(g.a == 42);
    assert(g.s == "hello");
}

int main() {
    std::thread p(producer);
    std::thread s(consumer);
    p.join();
    s.join();
    
}
```

## 3. acq_rel

acq_rel等于release + acquire，保证前面的读写不会排到下面，后面的读写不会排到上面，用于那些既需要读也需要写的操作

例如引用计数的减少，在决定是否释放的时候要确保之前的操作都完成了，以及后续的操作不会被重排

```cpp
#include <atomic>

template <class T>
class RefCounted {
    std::atomic<int> cnt_{1};
    T* ptr_;
public:
    explicit RefCounted(T* p) : ptr_(p) {}

    void addRef() {
        // 增加：relaxed 足够。因为持有引用本身就说明对象已经可见
        cnt_.fetch_add(1, std::memory_order_relaxed);
    }

    void release() {
        // 减少：必须 acq_rel
        //   release 半边 —— 本线程对对象的修改要对"最后那个人"可见
        //   acquire 半边 —— 本线程要看到别人此前的修改，才能安全析构
        if (cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete ptr_;
        }
    }
};

```

## seq_cst

## compare_exchange_weak

`compare_exchange_weak` 允许**伪失败**：即使原子变量的当前值等于
`expected`，操作也可能返回 `false`。因此它通常放在循环中使用。

下面通过 CAS 将计数器加一，但计数器最大不能超过 100：

```cpp
#include <atomic>

std::atomic<int> counter{0};

bool increment_if_below_100() {
    int expected = counter.load(std::memory_order_relaxed);

    while (expected < 100) {
        // 成功：counter 被写成 expected + 1，并返回 true。
        // 失败：counter 不变，expected 被自动更新为 counter 的当前值。
        if (counter.compare_exchange_weak(
                expected,
                expected + 1,
                std::memory_order_relaxed)) {
            return true;
        }

        // 无论是其他线程抢先修改，还是 weak 发生伪失败，都会重新尝试。
        // 不需要手动执行 expected = counter.load()。
    }

    return false;
}
```

例如 `expected` 原来为 5，但另一个线程先把 `counter` 改成了 8，本次
CAS 会失败，并把 `expected` 更新为 8。下一轮尝试写入的值就是 9。

## compare_exchange_strong

`compare_exchange_strong` 不允许伪失败。只要当前值确实等于
`expected`，比较交换就会成功，因此适合只尝试一次的状态转换。

```cpp
#include <atomic>

enum class State {
    idle,
    running,
    stopped
};

std::atomic<State> state{State::idle};

bool try_start() {
    State expected = State::idle;

    // 只允许将 idle 转换成 running，并且只尝试一次。
    if (state.compare_exchange_strong(expected, State::running)) {
        return true;  // 转换成功，当前线程取得启动权
    }

    // 转换失败说明状态确实不是 idle。
    // expected 已被改写为失败时观察到的实际状态。
    return false;
}
```

如果多个线程同时调用 `try_start()`，只有一个线程能把状态从 `idle`
改成 `running`。其他线程会失败，并在 `expected` 中得到它们观察到的
实际状态。

二者的选择原则：

- CAS 本来就在重试循环里：通常使用 `compare_exchange_weak`；
- 只尝试一次，失败后立即执行其他逻辑：通常使用
  `compare_exchange_strong`；
- `strong` 不是“线程竞争时一定成功”，当前值不等于 `expected` 时仍会失败；
- 两者失败时都会把原子变量的实际值写回 `expected`。
