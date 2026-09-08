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

## 