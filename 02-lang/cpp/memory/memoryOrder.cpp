// memoryOrder.cpp —— C++ 内存序的可运行示例
//
// 编译运行:
//   g++ -std=c++17 -O2 -pthread memoryOrder.cpp -o memoryOrder && ./memoryOrder
// 数据竞争检查:
//   g++ -std=c++17 -fsanitize=thread -g -O1 -pthread memoryOrder.cpp -o tsan && ./tsan
//
// 说明: demo1~demo5 分别对应 relaxed / release-acquire / acq_rel / seq_cst / 弱序反例

#include <atomic>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// ============================================================
// 1. relaxed —— 只要原子性, 不要顺序
//    场景: 统计计数器、profiling 打点、唯一 ID 分配
// ============================================================
namespace demo_relaxed {

std::atomic<long> counter{0};

void worker(int n) {
    for (int i = 0; i < n; ++i) {
        // 只需要"不丢计数", 不需要和任何其他内存操作定序
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}

void run() {
    constexpr int kThreads = 8, kIters = 100000;
    std::vector<std::thread> ts;
    for (int i = 0; i < kThreads; ++i) ts.emplace_back(worker, kIters);
    for (auto& t : ts) t.join();

    long got = counter.load(std::memory_order_relaxed);
    assert(got == static_cast<long>(kThreads) * kIters);
    std::printf("[relaxed]        counter = %ld (期望 %d) OK\n", got, kThreads * kIters);
}

}  // namespace demo_relaxed

// ============================================================
// 2. release / acquire —— 发布-订阅, 最常用
//    场景: 生产者-消费者、懒加载单例、自旋锁
// ============================================================
namespace demo_rel_acq {

struct Payload {
    int a;
    std::string s;
};

Payload payload;                   // 普通变量, 非原子
std::atomic<bool> ready{false};    // 只作为"同步旗子"

void producer() {
    payload.a = 42;                                    // (1)
    payload.s = "hello";                               // (2)
    ready.store(true, std::memory_order_release);      // (3) (1)(2) 不会重排到它之后
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) {    // (4) 之后的读不会提到它之前
        std::this_thread::yield();
    }
    // (3) release 与 (4) acquire 在同一个变量 ready 上配对, 建立 happens-before
    assert(payload.a == 42);
    assert(payload.s == "hello");
}

void run() {
    std::thread p(producer), c(consumer);
    p.join();
    c.join();
    std::printf("[release/acquire] payload = {%d, \"%s\"} OK\n", payload.a, payload.s.c_str());
}

}  // namespace demo_rel_acq

// ------------------------------------------------------------
// 2b. 自旋锁: lock 用 acquire, unlock 用 release
// ------------------------------------------------------------
namespace demo_spinlock {

class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock() {
        // acquire: 临界区里的读写不会被提到 lock 之前
        while (flag_.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();  // PAUSE 指令, 降低总线争用
#endif
        }
    }
    void unlock() {
        // release: 临界区里的写全部对下一个 lock 成功的线程可见
        flag_.clear(std::memory_order_release);
    }
};

SpinLock lk;
long shared = 0;  // 由 lk 保护的普通变量

void run() {
    constexpr int kThreads = 8, kIters = 50000;
    std::vector<std::thread> ts;
    for (int i = 0; i < kThreads; ++i) {
        ts.emplace_back([] {
            for (int j = 0; j < kIters; ++j) {
                lk.lock();
                ++shared;  // 非原子操作, 靠锁的 acquire/release 保证正确
                lk.unlock();
            }
        });
    }
    for (auto& t : ts) t.join();
    assert(shared == static_cast<long>(kThreads) * kIters);
    std::printf("[spinlock]        shared  = %ld (期望 %d) OK\n", shared, kThreads * kIters);
}

}  // namespace demo_spinlock

// ============================================================
// 3. acq_rel —— 用于 read-modify-write
//    场景: 引用计数递减、无锁栈/队列的 CAS
// ============================================================
namespace demo_acq_rel {

// 3a. 引用计数 (shared_ptr 的经典写法)
struct Object {
    int payload = 0;
    ~Object() { /* 析构必须能看到其他线程对 payload 的写 */ }
};

class RefCounted {
    std::atomic<int> cnt_{1};
    Object* ptr_;

public:
    explicit RefCounted(Object* p) : ptr_(p) {}

    void addRef() {
        // 增加: relaxed 足够。能调用 addRef 本身就说明对象已经对本线程可见
        cnt_.fetch_add(1, std::memory_order_relaxed);
    }

    void release() {
        // 减少: 必须 acq_rel
        //   release 半边 —— 本线程对对象的修改要对"最后那个人"可见
        //   acquire 半边 —— 本线程要看到别人此前的修改, 才能安全析构
        if (cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete ptr_;
            ptr_ = nullptr;
        }
    }
    Object* get() const { return ptr_; }
};

// 3b. 无锁栈 push
struct Node {
    int val;
    Node* next;
};
std::atomic<Node*> head{nullptr};

void push(int v) {
    Node* n = new Node{v, nullptr};
    n->next = head.load(std::memory_order_relaxed);
    // 成功: acq_rel —— release 发布 n 的构造, acquire 看到别的线程 push 的节点
    // 失败: relaxed  —— 失败只是重读 n->next, 下一轮循环还会再同步
    while (!head.compare_exchange_weak(n->next, n, std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
    }
}

void run() {
    // 引用计数
    auto* rc = new RefCounted(new Object{7});
    constexpr int kThreads = 8;
    std::vector<std::thread> ts;
    for (int i = 0; i < kThreads; ++i) {
        rc->addRef();
        ts.emplace_back([rc] { rc->release(); });
    }
    for (auto& t : ts) t.join();
    rc->release();  // 最后一次, 触发 delete
    assert(rc->get() == nullptr);
    delete rc;

    // 无锁栈
    ts.clear();
    for (int i = 0; i < kThreads; ++i) {
        ts.emplace_back([i] {
            for (int j = 0; j < 1000; ++j) push(i * 1000 + j);
        });
    }
    for (auto& t : ts) t.join();

    int n = 0;
    for (Node* p = head.load(std::memory_order_acquire); p;) {
        Node* nx = p->next;
        delete p;
        p = nx;
        ++n;
    }
    head.store(nullptr, std::memory_order_relaxed);
    assert(n == kThreads * 1000);
    std::printf("[acq_rel]         refcount 释放正确, 无锁栈节点 = %d OK\n", n);
}

}  // namespace demo_acq_rel

// ============================================================
// 4. seq_cst —— 需要跨多个原子变量的全局顺序
//    场景: Dekker / store-load 模式, 两个原子变量之间要求全局一致顺序
//    这是 seq_cst 唯一无法用 acq_rel 替代的地方: 它禁止 StoreLoad 重排
// ============================================================
namespace demo_seq_cst {

std::atomic<bool> x{false}, y{false};
std::atomic<int> z{0};

void write_x() { x.store(true, std::memory_order_seq_cst); }
void write_y() { y.store(true, std::memory_order_seq_cst); }

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)) {
    }
    if (y.load(std::memory_order_seq_cst)) ++z;
}
void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)) {
    }
    if (x.load(std::memory_order_seq_cst)) ++z;
}

void run() {
    // 如果把上面全部换成 release/acquire, z 有可能为 0:
    // x 和 y 之间没有任何同步关系, 两个读线程可以对"谁先"有不同的观察。
    // seq_cst 强制存在全局顺序, 于是至少一个读线程会看到两个都为 true。
    for (int round = 0; round < 200; ++round) {
        x = false;
        y = false;
        z = 0;
        std::thread a(write_x), b(write_y), c(read_x_then_y), d(read_y_then_x);
        a.join();
        b.join();
        c.join();
        d.join();
        assert(z.load() != 0);
    }
    std::printf("[seq_cst]         200 轮 Dekker 模式, z 恒不为 0 OK\n");
}

}  // namespace demo_seq_cst

// ============================================================
// 5. 反例: relaxed 不能用来发布数据
//    在 x86(TSO) 上几乎测不出问题, 在 ARM/POWER 上会真的挂。
//    这里只统计"读到旗子但数据还是旧值"的次数, 不 assert。
//    注意: 这段代码本身含 data race, TSan 会报警 —— 这正是它想演示的东西。
// ============================================================
namespace demo_broken {

int data = 0;
std::atomic<bool> ready{false};
std::atomic<int> anomalies{0};

void run() {
    constexpr int kRounds = 2000;
    for (int i = 0; i < kRounds; ++i) {
        data = 0;
        ready.store(false, std::memory_order_relaxed);

        std::thread p([] {
            data = 42;
            // ❌ relaxed store 不构成 release, data=42 可能被重排到它之后
            ready.store(true, std::memory_order_relaxed);
        });
        std::thread c([] {
            while (!ready.load(std::memory_order_relaxed)) {
            }
            // ❌ 没有 happens-before, 这里读到的 data 可能还是 0
            if (data != 42) anomalies.fetch_add(1, std::memory_order_relaxed);
        });
        p.join();
        c.join();
    }
    std::printf("[反例/relaxed]    %d 轮中观察到 %d 次数据不可见 (x86 上通常为 0, ARM 上可能 >0)\n",
                kRounds, anomalies.load());
}

}  // namespace demo_broken

int main() {
    demo_relaxed::run();
    demo_rel_acq::run();
    demo_spinlock::run();
    demo_acq_rel::run();
    demo_seq_cst::run();
    demo_broken::run();
    std::printf("\n全部 demo 通过。\n");
    return 0;
}
