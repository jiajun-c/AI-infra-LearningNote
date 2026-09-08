// 04_atomic_basics.cpp
//
// Lesson 04: std::atomic
//   - atomic 是无锁的, 但不是万能的
//   - load / store / fetch_add / compare_exchange
//   - 默认 memory_order_seq_cst 最安全, 但 acquire/release 更常用
//   - 复合操作必须用 CAS (compare_exchange), 不能 read-then-write
//
// 编译: g++ -std=c++17 -O2 -pthread 04_atomic_basics.cpp -o /tmp/04
// 运行: /tmp/04

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

// ---------- 例 1: atomic 计数器 ----------

static std::atomic<int> counter{0};

void increment_atomic() {
  for (int i = 0; i < 100000; ++i) {
    counter.fetch_add(1, std::memory_order_relaxed);  // 计数器用 relaxed 即可
  }
}

void demo_atomic_counter() {
  cout << "=== demo 1: atomic 计数器 ===\n";
  counter.store(0);
  std::thread a(increment_atomic);
  std::thread b(increment_atomic);
  a.join();
  b.join();
  cout << "期望 200000, 实际 " << counter.load() << "\n\n";
}

// ---------- 例 2: flag + acquire/release ----------

static std::atomic<bool> ready{false};
static int payload = 0;

void producer_rel() {
  payload = 123;  // 普通写
  ready.store(true, std::memory_order_release);  // release: 之前的写都对别人可见
}

void consumer_acq() {
  // acquire: 之后的读都看到 release 之前的写
  while (!ready.load(std::memory_order_acquire)) {
    // 自旋等
  }
  cout << "[consumer] payload = " << payload << " (期望 123)\n";
}

void demo_acq_rel() {
  cout << "=== demo 2: acquire / release flag ===\n";
  ready.store(false);
  payload = 0;
  std::thread p(producer_rel);
  std::thread c(consumer_acq);
  p.join();
  c.join();
  cout << "\n";
}

// ---------- 例 3: CAS 实现 lock-free counter ----------
//
// fetch_add 已经是硬件原语, 这里 CAS 是为了演示:
// 如果不用 fetch_add, 单纯 read-then-write 在多线程下会丢更新。
// compare_exchange_weak 会自动重试, 直到成功。

static std::atomic<int> cas_counter{0};

void increment_cas() {
  for (int i = 0; i < 100000; ++i) {
    int expected = cas_counter.load(std::memory_order_relaxed);
    while (!cas_counter.compare_exchange_weak(
        expected, expected + 1,
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
      // expected 已经被 CAS 改成最新值, 继续重试
    }
  }
}

void demo_cas() {
  cout << "=== demo 3: CAS 自实现原子加 ===\n";
  cas_counter.store(0);
  std::thread a(increment_cas);
  std::thread b(increment_cas);
  a.join();
  b.join();
  cout << "期望 200000, 实际 " << cas_counter.load() << "\n\n";
}

// ---------- 例 4: 错误示范 ----------
//
// 试图对 atomic 做 "if x == 5 then x = 10", 但读和写是分开的,
// 在另一个线程同时改 x 的情况下, 可能两个线程都看到 x==5, 都改,
// 写覆盖写, 丢更新。
// 正确做法是 compare_exchange。

// static std::atomic<int> x{5};
// void race_atomic() {
//   if (x.load() == 5) {   // <-- 读
//     x.store(10);         // <-- 写
//   }
// }

int main() {
  demo_atomic_counter();
  demo_acq_rel();
  demo_cas();
  return 0;
}