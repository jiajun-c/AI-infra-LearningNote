# Lock-free 算法

## 1. CAS自旋锁

不断检查一个位置的值是否为我们所需要的值，然后进行下一步的操作。

```cpp
class CASLock {
    std::atomic<bool> locked{false};
public:
    void lock() {
        while (locked.exchange(true, std::memory_order_acquire)) {
            // 自旋 / 退避
        }
    }
    void unlock() {
        locked.store(false, std::memory_order_release);
    }
};
```

缺点在于所有核在争抢一个cacheline -> cache line bouncing: 性能太差，同时

## 2. Ticket Lock

Ticket Lock相比与CAS而言更加的公平，避免了频率更高的线程一直抢占数据，如下所示。

```cpp
class TicketLock {
    std::atomic<uint64_t> next_{0}, now_{0};
public:
    void lock() {
        uint64_t t = next_.fetch_add(1, std::memory_order_relaxed);
        while (now_.load(std::memory_order_acquire) != t) _mm_pause();
    }
    void unlock() {
        now_.fetch_add(1, std::memory_order_release);
    }
};
```

## 3. MCS Lock

核心思想：每个线程在自己的本地变量上自旋，不抢同一个cacheline，同时使用分布式的链表去同步全局的状态，假设链表上有节点，表示当前有线程在持有锁

```cpp

```