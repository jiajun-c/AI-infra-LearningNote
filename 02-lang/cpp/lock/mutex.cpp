// mutex.cpp
//
// C++ 锁本体 (mutex 家族) 的实测 demo
//   - 6 种 mutex 类型的 sizeof
//   - 无竞争 / 有竞争下的 lock+unlock 开销
//   - 每种类型的典型用法
//   - 几个"看起来对其实是 UB"的写法
//
// 编译: g++ -std=c++17 -O2 -pthread mutex.cpp -o /tmp/mutex_demo
// 运行: /tmp/mutex_demo

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

// 防止编译器把临界区优化掉
static volatile long g_sink = 0;

static double ns_per_op(Clock::time_point t0, Clock::time_point t1, long ops) {
  return std::chrono::duration<double, std::nano>(t1 - t0).count() /
         static_cast<double>(ops);
}

// ---------------------------------------------------------------------------
// 1. 各类型的 sizeof
// ---------------------------------------------------------------------------
static void demo_sizeof() {
  std::printf("=== 1. 各 mutex 类型的 sizeof (x86-64 / libstdc++) ===\n\n");
  std::printf("  %-34s %s\n", "类型", "sizeof");
  std::printf("  %-34s %s\n", "----------------------------------", "------");
  std::printf("  %-34s %zu\n", "std::mutex", sizeof(std::mutex));
  std::printf("  %-34s %zu\n", "std::recursive_mutex", sizeof(std::recursive_mutex));
  std::printf("  %-34s %zu\n", "std::timed_mutex", sizeof(std::timed_mutex));
  std::printf("  %-34s %zu\n", "std::recursive_timed_mutex",
              sizeof(std::recursive_timed_mutex));
  std::printf("  %-34s %zu\n", "std::shared_mutex", sizeof(std::shared_mutex));
  std::printf("  %-34s %zu\n", "std::shared_timed_mutex",
              sizeof(std::shared_timed_mutex));
  std::printf("\n");
  std::printf("  对比: sizeof(std::atomic<int>) = %zu, sizeof(int) = %zu\n",
              sizeof(std::atomic<int>), sizeof(int));
  std::printf("  这些类型都不可拷贝、不可移动, 因为它们持有平台相关句柄\n");
  std::printf("  (Linux 上就是 pthread_mutex_t, glibc 里固定 40 字节)\n\n");
}

// ---------------------------------------------------------------------------
// 2. 无竞争开销: 单线程反复 lock/unlock
// ---------------------------------------------------------------------------
template <typename Mutex>
static void bench_uncontended(const char* name, long iters) {
  Mutex m;
  long local = 0;
  auto t0 = Clock::now();
  for (long i = 0; i < iters; ++i) {
    m.lock();
    local += 1;      // 临界区: 一条加法
    m.unlock();
  }
  auto t1 = Clock::now();
  g_sink = local;
  std::printf("  %-30s %8.1f ns/op\n", name, ns_per_op(t0, t1, iters));
}

static void demo_uncontended() {
  std::printf("=== 2. 无竞争 lock+unlock 开销 (单线程, 1e7 次) ===\n\n");
  constexpr long kIters = 10'000'000;
  bench_uncontended<std::mutex>("std::mutex", kIters);
  bench_uncontended<std::recursive_mutex>("std::recursive_mutex", kIters);
  bench_uncontended<std::timed_mutex>("std::timed_mutex", kIters);
  bench_uncontended<std::shared_mutex>("std::shared_mutex (写锁)", kIters);
  std::printf("\n");
  std::printf("  无竞争时都是用户态原子操作, 不陷内核, 所以只差几 ns\n");
  std::printf("  递归锁贵在要比较 owner tid + 维护计数\n\n");
}

// ---------------------------------------------------------------------------
// 3. 有竞争开销: N 个线程抢同一把锁
// ---------------------------------------------------------------------------
template <typename Mutex>
static void bench_contended(const char* name, int nthreads, long per_thread) {
  Mutex m;
  long counter = 0;
  std::vector<std::thread> ts;
  ts.reserve(nthreads);

  auto t0 = Clock::now();
  for (int i = 0; i < nthreads; ++i) {
    ts.emplace_back([&] {
      for (long j = 0; j < per_thread; ++j) {
        std::lock_guard<Mutex> lk(m);
        ++counter;
      }
    });
  }
  for (auto& t : ts) t.join();
  auto t1 = Clock::now();

  g_sink = counter;
  const long total = nthreads * per_thread;
  const double ns = ns_per_op(t0, t1, total);
  std::printf("  %-30s %8.1f ns/op  (吞吐 %6.2f M ops/s)\n", name, ns,
              1000.0 / ns);
}

static void demo_contended() {
  // 用 2 个线程制造"配对争抢"：一个持锁时另一个正好在等，
  // 最容易看清锁本身的开销。线程再多会退化成排队，反而看不清。
  constexpr int kThreads = 2;
  const long per_thread = 200'000;

  std::printf("=== 3. 有竞争 lock+unlock 开销 (%d 线程抢同一把锁) ===\n\n",
              kThreads);
  bench_contended<std::mutex>("std::mutex", kThreads, per_thread);
  bench_contended<std::recursive_mutex>("std::recursive_mutex", kThreads, per_thread);
  bench_contended<std::shared_mutex>("std::shared_mutex (写锁)", kThreads, per_thread);
  std::printf("\n");
  std::printf("  (本机 hardware_concurrency = %u, 但用 2 线程才能看清单次开销)\n\n",
              std::thread::hardware_concurrency());

  // 对比: 多个读者并行 vs 多个写者串行
  std::printf("  读多写少场景对比 (4 线程, 各 200k 次):\n");
  {
    std::shared_mutex rw;
    long acc = 0;
    auto t0 = Clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) {
      ts.emplace_back([&] {
        for (long j = 0; j < per_thread; ++j) {
          std::shared_lock<std::shared_mutex> lk(rw);   // 读锁, 可并行
          acc += 1;
        }
      });
    }
    for (auto& t : ts) t.join();
    auto t1 = Clock::now();
    g_sink = acc;
    std::printf("    %-28s %8.1f ns/op\n", "4 读者 (shared_lock)",
                ns_per_op(t0, t1, 4 * per_thread));
  }
  {
    std::shared_mutex rw;
    long acc = 0;
    auto t0 = Clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) {
      ts.emplace_back([&] {
        for (long j = 0; j < per_thread; ++j) {
          std::lock_guard<std::shared_mutex> lk(rw);    // 写锁, 串行
          acc += 1;
        }
      });
    }
    for (auto& t : ts) t.join();
    auto t1 = Clock::now();
    g_sink = acc;
    std::printf("    %-28s %8.1f ns/op\n", "4 写者 (unique_lock)",
                ns_per_op(t0, t1, 4 * per_thread));
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// 4. 各类型用法示例
// ---------------------------------------------------------------------------
static void demo_usage() {
  std::printf("=== 4. 各类型用法 ===\n\n");

  // --- mutex: 默认选择 ---
  {
    std::mutex m;
    std::lock_guard<std::mutex> lk(m);   // RAII, 析构自动 unlock
    g_sink = 1;
    std::printf("  [mutex]          lock_guard 就够了\n");
  }

  // --- try_lock: 拿不到就去做别的事 ---
  {
    std::mutex m;
    if (m.try_lock()) {
      std::printf("  [mutex]          try_lock 拿到锁\n");
      m.unlock();
    } else {
      std::printf("  [mutex]          try_lock 没拿到, 走别的路径\n");
    }
    // 注意: try_lock 允许"虚假失败" —— 即使没人持锁也可能返回 false
  }

  // --- recursive_mutex: 同线程重入 ---
  {
    std::recursive_mutex rm;
    struct Rec {
      std::recursive_mutex& rm;
      void a() {
        std::lock_guard<std::recursive_mutex> lk(rm);
        b();                                  // 内部又加锁, 不会死锁
      }
      void b() { std::lock_guard<std::recursive_mutex> lk(rm); }
    };
    Rec r{rm};
    r.a();
    std::printf("  [recursive]      同线程重入两次, 计数归零\n");
  }

  // --- timed_mutex: 带超时 ---
  {
    std::timed_mutex tm;
    bool got = tm.try_lock_for(std::chrono::milliseconds(10));
    if (got) {
      std::printf("  [timed]          try_lock_for(10ms) 成功\n");
      tm.unlock();
    }
    // 也可以 try_lock_until(deadline)
  }

  // --- shared_mutex: 读写分离 ---
  {
    std::shared_mutex rw;
    int value = 0;
    {
      std::unique_lock<std::shared_mutex> wlk(rw);   // 写: 独占
      value = 42;
    }
    {
      std::shared_lock<std::shared_mutex> rlk(rw);   // 读: 可共享
      std::printf("  [shared_mutex]   读到 value = %d\n", value);
    }
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// 5. 反例: 这些写法是 UB 或必然死锁 (默认注释掉, 想验证可单独放开)
// ---------------------------------------------------------------------------
static void demo_pitfalls_note() {
  std::printf("=== 5. 常见坑 ===\n\n");
  std::printf("  a) 忘记 unlock / 中途 return   → 死锁   (用 RAII 解决)\n");
  std::printf("  b) 非递归 mutex 同线程重复 lock → 死锁   (见下)\n");
  std::printf("  c) unlock 一把没持有的锁        → UB\n");
  std::printf("  d) 析构时仍被 lock              → UB\n");
  std::printf("  e) try_lock 返回 false 不等于有人持锁 (允许虚假失败)\n");
  std::printf("  f) 标准不保证公平, 高竞争下可能饿死某个线程\n");
  std::printf("\n");

  // b) 的演示: 用 try_lock 安全地展示"重复 lock 会失败"
  std::mutex m;
  m.lock();
  bool again = m.try_lock();
  std::printf("  b) 同一线程对 std::mutex 重复加锁: try_lock 返回 %s\n",
              again ? "true (不该发生)" : "false → 换成 lock() 就是死锁");
  m.unlock();
  std::printf("     (std::recursive_mutex 在这里会返回 true)\n\n");
}

int main() {
  demo_sizeof();
  demo_uncontended();
  demo_contended();
  demo_usage();
  demo_pitfalls_note();
  return 0;
}
