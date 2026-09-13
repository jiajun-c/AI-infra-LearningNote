// bench_cas.cpp - 测试 CAS 自旋锁在不同场景下的性能短板
//
// 编译: g++ -O2 -std=c++17 -pthread bench_cas.cpp -o bench_cas
// 运行: ./bench_cas
//
// 测试维度:
//   1. 临界区大小对吞吐的影响 (空锁 vs 重锁)
//   2. 线程数扩展性 (1 -> N核)
//   3. 高竞争 vs 低竞争
//   4. CAS 退避策略 (无退避 / 指数退避)
//   5. 对比 std::mutex (作为基线)

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <string>
#include <functional>
#include <cmath>

using namespace std::chrono_literals;

// ============= 各种锁实现 =============

// 1. 无退避的 CAS 自旋锁 (cas.cpp 中的原始实现)
class CASLockNoBackoff {
    std::atomic<bool> locked{false};
public:
    void lock() {
        while (locked.exchange(true, std::memory_order_acquire)) {
            // 紧贴自旋, 无退避 -> 缓存行乒乓
        }
    }
    void unlock() {
        locked.store(false, std::memory_order_release);
    }
};

// 2. 带指数退避的 CAS 自旋锁
class CASLockExpBackoff {
    std::atomic<bool> locked{false};
public:
    void lock() {
        // 先尝试一次, 失败再退避自旋
        if (locked.exchange(true, std::memory_order_acquire)) {
            int spins = 0;
            while (locked.exchange(true, std::memory_order_acquire)) {
                // 指数退避: 1, 2, 4, 8, ... 最多 1024
                if (spins < 10) {
                    int delay = 1 << (spins < 6 ? spins : 6); // 最多 64 次循环
                    for (int i = 0; i < delay; ++i) {
                        __builtin_ia32_pause();
                    }
                    ++spins;
                } else {
                    // 重竞争时偶尔让出 CPU
                    std::this_thread::yield();
                    spins = 5; // 重置, 避免长时间空转
                }
            }
        }
    }
    void unlock() {
        locked.store(false, std::memory_order_release);
    }
};

// 3. std::mutex 作为对照
using StdMutex = std::mutex;

// ============= 基准测试框架 =============

struct BenchResult {
    std::string name;
    int threads;
    int critical_work;     // 临界区内模拟的工作量 (次空操作)
    int total_iters;       // 每个线程总迭代次数
    double elapsed_ms;     // 总耗时 ms
    double ops_per_sec;    // 每秒 lock/unlock 对数
};

template <typename Lock, typename CriticalWork>
BenchResult run_bench(const std::string& name, int threads, int critical_work,
                      int total_iters, CriticalWork work) {
    Lock lk;
    std::atomic<bool> start{false};
    std::vector<std::thread> ts;

    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            // 每个线程独立计数, 避免原子竞争污染测量
            int local_iters = total_iters;
            for (int i = 0; i < local_iters; ++i) {
                lk.lock();
                work(); // 临界区
                lk.unlock();
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& th : ts) th.join();
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ops = static_cast<double>(threads) * total_iters;
    BenchResult r{name, threads, critical_work, total_iters, ms, ops / (ms / 1000.0)};
    return r;
}

// 简单的"工作"函数: 空循环次数
inline void do_work(int n) {
    volatile long x = 0;
    for (int i = 0; i < n; ++i) x += i;
}

// ============= 报告打印 =============

void print_header() {
    std::cout << std::string(110, '=') << "\n";
    std::cout << "CAS 自旋锁性能基准 - 暴露短板场景\n";
    std::cout << std::string(110, '=') << "\n";
    std::cout << "硬件线程数: "
              << std::thread::hardware_concurrency() << "\n\n";
}

void print_result(const BenchResult& r) {
    std::cout << std::left
              << std::setw(28) << r.name
              << std::setw(8)  << r.threads
              << std::setw(12) << r.critical_work
              << std::setw(10) << r.total_iters
              << std::setw(12) << std::fixed << std::setprecision(2) << r.elapsed_ms
              << std::setw(16) << std::setprecision(0) << r.ops_per_sec
              << "\n";
}

void print_table_header() {
    std::cout << std::left
              << std::setw(28) << "锁类型"
              << std::setw(8)  << "线程"
              << std::setw(12) << "临界工作量"
              << std::setw(10) << "迭代/线程"
              << std::setw(12) << "耗时(ms)"
              << std::setw(16) << "ops/s"
              << "\n";
    std::cout << std::string(86, '-') << "\n";
}

// ============= 实验场景 =============

// 实验 1: 临界区为空 -> 暴露 CAS 紧密自旋导致的缓存行抖动
void exp_empty_critical(int hw_threads) {
    std::cout << "\n[实验1] 临界区为空 (CAS 紧密自旋的缓存行抖动)\n";
    std::cout << "  目的: 每次 lock/unlock 之间几乎没有工作, 线程在 CAS 上空转 -> \n";
    std::cout << "        所有线程反复抢占同一个缓存行, 吞吐严重下降\n\n";
    print_table_header();

    const int iters = 200000;
    for (int n : {1, 2, 4, std::min(hw_threads, 8)}) {
        if (n > hw_threads * 2) break;
        print_result(run_bench<CASLockNoBackoff>(
            "CAS(无退避)", n, 0, iters, []{ }));
        print_result(run_bench<CASLockExpBackoff>(
            "CAS(指数退避)", n, 0, iters, []{ }));
        print_result(run_bench<StdMutex>(
            "std::mutex",   n, 0, iters, []{ }));
    }
}

// 实验 2: 临界区有适度工作 -> CAS 表现接近正常
void exp_medium_critical(int hw_threads) {
    std::cout << "\n[实验2] 临界区有适度工作 (~100 次循环)\n";
    std::cout << "  目的: 当临界区足够长, 锁被持有的时间 >> 自旋开销\n";
    std::cout << "        退避收益变小, 但 CAS 仍优于 mutex\n\n";
    print_table_header();

    const int iters = 50000;
    for (int n : {1, 2, 4, std::min(hw_threads, 8)}) {
        if (n > hw_threads * 2) break;
        print_result(run_bench<CASLockNoBackoff>(
            "CAS(无退避)", n, 100, iters, []{ do_work(100); }));
        print_result(run_bench<CASLockExpBackoff>(
            "CAS(指数退避)", n, 100, iters, []{ do_work(100); }));
        print_result(run_bench<StdMutex>(
            "std::mutex",   n, 100, iters, []{ do_work(100); }));
    }
}

// 实验 3: 临界区非常长 -> 锁争用不再是瓶颈
void exp_long_critical(int hw_threads) {
    std::cout << "\n[实验3] 临界区很长 (~5000 次循环)\n";
    std::cout << "  目的: 当临界区很长, 自旋锁的吞吐完全被单线程工作决定\n";
    std::cout << "        退避几乎无影响, 但单线程被串行化\n\n";
    print_table_header();

    const int iters = 5000;
    for (int n : {1, 2, 4, std::min(hw_threads, 8)}) {
        if (n > hw_threads * 2) break;
        print_result(run_bench<CASLockNoBackoff>(
            "CAS(无退避)", n, 5000, iters, []{ do_work(5000); }));
        print_result(run_bench<CASLockExpBackoff>(
            "CAS(指数退避)", n, 5000, iters, []{ do_work(5000); }));
        print_result(run_bench<StdMutex>(
            "std::mutex",   n, 5000, iters, []{ do_work(5000); }));
    }
}

// 实验 4: 跨 NUMA / 多核扩展性
void exp_scaling(int /*hw_threads*/) {
    std::cout << "\n[实验4] 扩展性: 1 -> 16 线程, 临界区为空\n";
    std::cout << "  目的: 观察 CAS 自旋锁的吞吐量随线程数如何变化\n";
    std::cout << "        理论上: 加锁数量翻倍, 但吞吐应基本持平或下降\n";
    std::cout << "        实际: 缓存行争用会导致吞吐骤降\n\n";
    print_table_header();

    const int iters = 100000;
    for (int n : {1, 2, 4, 8, 16}) {
        print_result(run_bench<CASLockNoBackoff>(
            "CAS(无退避)", n, 0, iters, []{ }));
        print_result(run_bench<CASLockExpBackoff>(
            "CAS(指数退避)", n, 0, iters, []{ }));
        print_result(run_bench<StdMutex>(
            "std::mutex",   n, 0, iters, []{ }));
    }
}

// 实验 5: 总结
void summary() {
    std::cout << "\n" << std::string(110, '=') << "\n";
    std::cout << "CAS 自旋锁性能短板总结\n";
    std::cout << std::string(110, '=') << "\n";
    std::cout << R"(
[短板 1] 缓存行抖动 (cache line bouncing)
  - 紧贴自旋的 CAS 会让所有等待线程在同一缓存行上反复 RMW
  - 即使没拿到锁, 每次 exchange 都强制 invalidate 其它核心的缓存
  - 表现: 多核下吞吐比单核差, 而不是线性扩展

[短板 2] 空临界区表现差
  - 临界区越短, 自旋开销占总耗时比例越大
  - 单次 lock/unlock 在 ns 量级, 自旋重试却要 ~10-100 ns
  - 表现: 空临界区下 CAS 吞吐 < std::mutex (mutex 在内核态 park)

[短板 3] 公平性问题
  - CAS 自旋锁不保证公平性, 可能存在线程饥饿
  - 高争用下某些线程可能一直抢不到锁

[短板 4] 优先级反转
  - 低优先级线程持锁, 高优先级线程空转占满 CPU
  - 系统调度器无法介入

[短板 5] 功耗浪费
  - 紧贴自旋让 CPU 始终跑在高频, 浪费电力
  - 退避 + yield 可以缓解但引入额外延迟

[何时使用 CAS 自旋锁]
  ✓ 临界区极短 (< 几十 ns) 且线程数受控
  ✓ 持有锁时绝不会调度/阻塞
  ✓ 实现简单, 无系统调用开销
  ✗ 多核高争用
  ✗ 临界区不可预测 (可能持有较久)
  ✗ 需要公平性
)";
}

int main() {
    int hw = std::max(1u, std::thread::hardware_concurrency());
    print_header();

    exp_empty_critical(hw);
    exp_medium_critical(hw);
    exp_long_critical(hw);
    exp_scaling(hw);
    summary();

    return 0;
}
