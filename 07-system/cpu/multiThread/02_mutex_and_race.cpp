// 02_mutex_and_race.cpp
//
// Lesson 02: std::mutex 与数据竞争
//   - 数据竞争 (data race) 是未定义行为
//   - mutex + lock_guard 是默认选择
//   - 多把 mutex 用 scoped_lock 避免死锁
//   - unique_lock 的灵活性
//
// 编译: g++ -std=c++17 -O2 -pthread 02_mutex_and_race.cpp -o /tmp/02
// 运行: /tmp/02

#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;

// ---------- 例 1: 数据竞争 ----------
//
// 多个线程同时 ++counter 几乎肯定是错的。
// counter++ 实际上是 read-modify-write 三步，并发时互相覆盖。
//
// 跑两次看输出:
//   $ for i in 1 2 3 4 5; do /tmp/02; done
// 会发现每次结果不一样, 而且几乎都不等于 200000。

static int counter_racy = 0;

void increment_racy() {
  for (int i = 0; i < 100000; ++i) {
    counter_racy++;   // <-- data race
  }
}

void demo_race() {
  cout << "=== demo 1: data race ===\n";
  counter_racy = 0;
  std::thread a(increment_racy);
  std::thread b(increment_racy);
  a.join();
  b.join();
  cout << "期望 200000, 实际 " << counter_racy
       << "  ← 不一致就说明有 race\n\n";
}

// ---------- 例 2: 用 mutex 保护 ----------
//
// mutex 保护下的 counter++ 就是确定的。

static int counter_safe = 0;
static std::mutex mu_safe;

void increment_safe() {
  for (int i = 0; i < 100000; ++i) {
    std::lock_guard<std::mutex> g(mu_safe);  // RAII: 析构时自动 unlock
    counter_safe++;
  }
}

void demo_mutex() {
  cout << "=== demo 2: 加 mutex ===\n";
  counter_safe = 0;
  std::thread a(increment_safe);
  std::thread b(increment_safe);
  a.join();
  b.join();
  cout << "期望 200000, 实际 " << counter_safe << "\n\n";
}

// ---------- 例 3: scoped_lock 多把锁 ----------
//
// 两个账户互相转账: A->B 和 B->A 同时跑。
// 如果先锁 A 再锁 B, 反向操作先锁 B 再锁 A, 经典死锁。
// std::scoped_lock 会按内部算法统一加锁, 自动避开死锁。

struct Account {
  int balance = 1000;
  std::mutex mu;
};

void transfer(Account& from, Account& to, int amount) {
  std::scoped_lock lock(from.mu, to.mu);  // C++17, 多锁安全
  from.balance -= amount;
  to.balance   += amount;
}

void demo_scoped_lock() {
  cout << "=== demo 3: scoped_lock 避免死锁 ===\n";
  Account a, b;
  std::thread t1([&] { for (int i = 0; i < 1000; ++i) transfer(a, b, 1); });
  std::thread t2([&] { for (int i = 0; i < 1000; ++i) transfer(b, a, 1); });
  t1.join();
  t2.join();
  cout << "a.balance = " << a.balance << ", b.balance = " << b.balance
       << " (期望都是 1000)\n\n";
}

// ---------- 例 4: 死锁的反例 ----------
//
// 注释掉 scoped_lock, 改用手动 lock 两个 mutex,
// 顺序不一致时几乎必现死锁。

// std::mutex m1, m2;
//
// void bad1() { std::lock_guard<std::mutex> g(m1); usleep(1000);
//              std::lock_guard<std::mutex> g(m2); }
// void bad2() { std::lock_guard<std::mutex> g(m2); usleep(1000);
//              std::lock_guard<std::mutex> g(m1); }

int main() {
  demo_race();
  demo_mutex();
  demo_scoped_lock();
  return 0;
}