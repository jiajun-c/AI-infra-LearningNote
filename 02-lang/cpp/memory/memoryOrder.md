# C++ 内存序

C++的memory order大概有下面的几种

- relaxed：仅有原子性，但是不保证顺序一致性和可见性
- acquire：有原子性，同时保证之后的读不会重排到此之前，与释放该变量的写线程进行同步
- release：有原子性，之前的写不会重排到此之后，同时与获取该变量的读线程同步
- acq_rel: acquire+rel
- seq_cst: 全局顺序，全部线程看到一个顺序

配套可运行代码见 [memoryOrder.cpp](./memoryOrder.cpp)，编译：

```bash
g++ -std=c++17 -O2 -pthread memoryOrder.cpp -o memoryOrder && ./memoryOrder
```

## 0. 先记住选择原则

| 场景 | 选择 |
| --- | --- |
| 只要计数正确，不关心和其他数据的先后 | `relaxed` |
| 一个线程写数据 + 发布标志，另一个线程读标志 + 读数据 | `release` / `acquire` 配对 |
| 同一个原子变量上"既读又写"，且要同时看到前后的数据 | `acq_rel`（用于 RMW） |
| 多个原子变量之间需要一个全局统一的先后顺序 | `seq_cst` |

一句话：**同步的是「数据」，原子变量只是搬运顺序的载体**。release/acquire 建立的是 happens-before 边，让 release 之前写的**普通变量**对 acquire 之后可见。

---

## 1. relaxed：只要原子性，不要顺序

### relaxed 适用场景

- 统计计数器（QPS、命中次数、profiling 打点）
- `shared_ptr` 的引用计数**增加**（`fetch_add`）——增加时不需要看到别的数据
- 生成唯一 ID / 分配 ticket
- 无锁队列里"先用 relaxed 探一下"再用强内存序确认

### relaxed 代码

```cpp
#include <atomic>
#include <thread>
#include <vector>
#include <cassert>

std::atomic<long> g_counter{0};

void worker(int n) {
    for (int i = 0; i < n; ++i) {
        // 只需要"不丢计数"，不需要和任何其他内存操作定序
        g_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

int main() {
    std::vector<std::thread> ts;
    for (int i = 0; i < 8; ++i) ts.emplace_back(worker, 100000);
    for (auto& t : ts) t.join();
    // 结果一定是 800000：原子性保证了，只是"什么时候被别人看到"没保证
    assert(g_counter.load(std::memory_order_relaxed) == 800000);
}
```

### 反例：relaxed 不能用来发布数据

```cpp
int  data = 0;
std::atomic<bool> ready{false};

// 线程 1
data = 42;
ready.store(true, std::memory_order_relaxed);   // ❌ 可能被重排到 data=42 之前

// 线程 2
while (!ready.load(std::memory_order_relaxed)) {}
assert(data == 42);   // ❌ 可能失败（尤其在 ARM/POWER 这类弱内存模型上）
```

---

## 2. release / acquire：发布 - 订阅（最常用）

### release/acquire 适用场景

- **生产者-消费者**：写好数据后置位标志，消费者看到标志就能安全读数据
- **单次初始化 / 懒加载单例**：构造完对象后发布指针
- **自旋锁**：`lock` 用 acquire，`unlock` 用 release（临界区不会漏出去）
- 无锁队列的 head/tail 发布

### 代码：发布数据

```cpp
#include <atomic>
#include <thread>
#include <cassert>
#include <string>

struct Payload { int a; std::string s; };

Payload g_payload;                 // 普通变量，非原子
std::atomic<bool> g_ready{false};  // 只作为"同步旗子"

void producer() {
    g_payload.a = 42;                                   // (1)
    g_payload.s = "hello";                              // (2)
    g_ready.store(true, std::memory_order_release);     // (3) 屏障：(1)(2) 不会重排到它之后
}

void consumer() {
    while (!g_ready.load(std::memory_order_acquire)) {  // (4) 屏障：之后的读不会提到它之前
        std::this_thread::yield();
    }
    // (3) release 与 (4) acquire 配对，建立 happens-before
    assert(g_payload.a == 42);        // ✅ 一定成立
    assert(g_payload.s == "hello");   // ✅ 一定成立
}
```

### 代码：自旋锁
condition
```cpp
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() {
        // acquire：临界区里的读写不会被提到 lock 之前
        while (flag_.test_and_set(std::memory_order_acquire)) {
            #if defined(__x86_64__)
                __builtin_ia32_pause();   // PAUSE 指令，降低总线争用
            #endif
        }
    }
    void unlock() {
        // release：临界区里的写全部对下一个 lock 成功的线程可见
        flag_.clear(std::memory_order_release);
    }
};
```

### 注意：release/acquire 是**逐变量配对**的

只有对**同一个原子变量**的 release-store 和 acquire-load 才能建立同步关系。对 `x` 做 release、对 `y` 做 acquire，是没有任何关系的。

---

## 3. acq_rel：用于 read-modify-write（RMW）

`acq_rel` 只对 RMW 操作有意义（`fetch_add` / `exchange` / `compare_exchange_*`），因为它同时是读又是写：读的那半边是 acquire，写的那半边是 release。

### acq_rel 适用场景

- **引用计数的减少**：最后一个释放者必须看到其他线程之前所有的写，才能安全析构
- 无锁栈 / 队列的 CAS push/pop（既要把新节点发布出去，又要看到别人发布的节点）
- 多生产者队列的 slot 抢占

### 代码：引用计数（`shared_ptr` 的经典写法）

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

> 进阶优化（libstdc++ 就是这么做的）：`fetch_sub` 用 `release`，只有当它返回 1、真正要析构时，再补一个 `std::atomic_thread_fence(std::memory_order_acquire)`。这样常见路径省掉一次 acquire 开销。

### 代码：无锁栈 push

```cpp
struct Node { int val; Node* next; };
std::atomic<Node*> head{nullptr};

void push(int v) {
    Node* n = new Node{v, nullptr};
    n->next = head.load(std::memory_order_relaxed);
    // 成功：acq_rel —— release 发布 n 的构造，acquire 看到别的线程 push 的节点
    // 失败：relaxed  —— 失败只是重读 n->next，下一轮循环还会再同步
    while (!head.compare_exchange_weak(n->next, n,
                                       std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {}
}
```

---

## 4. seq_cst：需要跨多个原子变量的全局顺序

默认值（不写第二个参数就是 `seq_cst`）。它在 acq_rel 之上额外提供一条**所有 seq_cst 操作的单一全局顺序**，所有线程看到的这个顺序是一致的。

### seq_cst 适用场景

- **Dekker / store-load 模式**：两个线程各写一个变量再读对方的变量，要求"不能都读到旧值"
- 需要多个原子变量之间存在全局一致顺序的算法（Peterson 锁、某些 seqlock 变体）
- **不确定用什么的时候** —— 先用默认的 seq_cst 保证正确，profiling 之后再放松

### 代码：只有 seq_cst 能保证的场景

```cpp
#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x{false}, y{false};
std::atomic<int>  z{0};

```cpp
void write_x() { x.store(true, std::memory_order_seq_cst); }
void write_y() { y.store(true, std::memory_order_seq_cst); }

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)) {}
    if (y.load(std::memory_order_seq_cst)) ++z;
}
void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)) {}
    if (x.load(std::memory_order_seq_cst)) ++z;
}

int main() {
    std::thread a(write_x), b(write_y), c(read_x_then_y), d(read_y_then_x);
    a.join(); b.join(); c.join(); d.join();
    assert(z.load() != 0);   // ✅ seq_cst 下一定成立
                             // ❌ 如果全部换成 acq_rel/release-acquire，z 可能为 0
}
```

原因：release/acquire 只保证**同一个变量**上的配对同步，`x` 和 `y` 之间没有任何关系，两个读线程完全可以对"x 先还是 y 先"有不同的观察。seq_cst 强制存在一个全局顺序，`x` 和 `y` 必有一个先发生，于是至少一个读线程会看到两个都为 true。

这也是 seq_cst 唯一无法用 acq_rel 替代的地方：它禁止了 **StoreLoad 重排**。

---

## 5. memory_order_consume（了解即可）

原本设计给"数据依赖"场景（如 RCU 读侧：读到指针后只访问该指针指向的数据），比 acquire 更便宜。但因为编译器难以实现依赖链追踪，**所有主流编译器都把它当 acquire 处理**，C++17 起标准也建议不要使用。写代码直接用 `acquire`。

---

## 6. 硬件成本：为什么 x86 上感觉不出差别

| 内存序 | x86-64 生成的指令 | ARM64 生成的指令 |
| --- | --- | --- |
| relaxed load/store | `mov` / `mov` | `ldr` / `str` |
| acquire load | `mov`（免费） | `ldar` |
| release store | `mov`（免费） | `stlr` |
| seq_cst store | `xchg` 或 `mov + mfence`（**贵**） | `stlr` + `dmb ish` |

x86-64 是 TSO 模型，硬件本身就不会做 StoreStore / LoadLoad / LoadStore 重排，**只会重排 StoreLoad**。所以：

- 在 x86 上，`relaxed` / `acquire` / `release` 生成的机器码几乎一样，差别只体现在**编译器**是否敢重排。
- 唯一有实打实开销的是 `seq_cst` 的 store（需要 `mfence`/`xchg` 来禁止 StoreLoad 重排）。
- **在 x86 上测不出 bug ≠ 代码正确**。同样的代码放到 ARM（Apple Silicon、Graviton、手机）上就可能挂。写弱内存序代码务必用 TSan / ARM 机器验证。

```bash
# 用 ThreadSanitizer 检查数据竞争
g++ -std=c++17 -fsanitize=thread -g -O1 -pthread memoryOrder.cpp -o tsan_test && ./tsan_test
```

---

## 7. 常见坑

1. **以为 relaxed 只是"慢一点但正确"** —— 不，它会导致真正的逻辑错误。
2. **release/acquire 不配对** —— 对不同变量做 release/acquire 没有任何同步效果。
3. **在 `while` 轮询里用 relaxed** —— 编译器可能把 load 提到循环外，变成死循环。
4. **CAS 的失败内存序不能强于成功内存序** —— C++17 前是 UB，C++17 后要求 `failure <= success`。
5. **过早优化** —— 先全部用默认 `seq_cst` 写对，profiling 证明它是瓶颈了再逐处放松，每放松一处都要说清楚"它和哪个操作配对"。

## 参考

- cppreference: [std::memory_order](https://en.cppreference.com/w/cpp/atomic/memory_order)
- Herb Sutter, *atomic<> Weapons* (C++ and Beyond 2012)
- Paul E. McKenney, *Is Parallel Programming Hard...*
