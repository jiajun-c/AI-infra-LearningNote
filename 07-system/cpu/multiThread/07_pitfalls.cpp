// 07_pitfalls.cpp
//
// Lesson 07: 常见陷阱
//   这一节不是完整 demo, 是把大家最容易踩的坑用最小例子列出来,
//   每个 case 都用 注释 + 修复版本 的形式给出。
//
// 编译运行: 不直接编译 (里面大部分是有 bug 的代码)。
// 改成自己想跑哪个, 单独编译就行。

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;

// =============================================================
// 陷阱 1: 假唤醒 (spurious wakeup) 和 lost wakeup
// =============================================================
//
// 反例: 不带 predicate 的 wait
//
// static std::mutex mu;
// static std::condition_variable cv;
// static bool ready = false;
//
// void producer_wrong() {
//   {
//     std::lock_guard<std::mutex> g(mu);
//     ready = true;
//   }
//   cv.notify_one();  // 可能在 consumer 还没 wait 时就 notify 了 → lost wakeup
// }
//
// void consumer_wrong() {
//   std::unique_lock<std::mutex> lk(mu);
//   cv.wait(lk);       // ← 即使 ready 是 true, 也可能继续睡过去
//   if (!ready) { ... } // 必须再检查
// }

// 修复版本
static std::mutex mu_ok;
static std::condition_variable cv_ok;
static bool ready_ok = false;

void producer_ok() {
  {
    std::lock_guard<std::mutex> g(mu_ok);
    ready_ok = true;
  }
  cv_ok.notify_one();
}

void consumer_ok() {
  std::unique_lock<std::mutex> lk(mu_ok);
  cv_ok.wait(lk, [] { return ready_ok; });  // ← 带 predicate, 自动 while 重检
  // 这里 ready_ok 保证是 true
  cout << "[陷阱 1] consumer 安全醒来, ready=" << ready_ok << "\n";
}

void demo_spurious_wakeup() {
  std::thread t1(producer_ok);
  std::thread t2(consumer_ok);
  t1.join();
  t2.join();
}

// =============================================================
// 陷阱 2: 死锁 — 手动 lock 两把 mutex 顺序不一致
// =============================================================
//
// 反例:
//   std::mutex a, b;
//   void f() { std::lock_guard g1(a); std::lock_guard g2(b); }
//   void g() { std::lock_guard g1(b); std::lock_guard g2(a); } // 顺序反了
//
// 修复: 用 std::scoped_lock(a, b), 让标准库按内部约定统一加锁,
//       或者永远按固定顺序加锁 (例如先锁地址小的)。

// =============================================================
// 陷阱 3: 在锁里 sleep / IO — 吞吐塌方
// =============================================================
//
// 反例:
//   std::unique_lock<std::mutex> lk(mu);
//   do_slow_io();   // 几 ms 甚至几秒, 其他线程全在等这把锁
//   do_thing();
//   lk.unlock();
//
// 修复: 把慢活挪出临界区。
//
//   WorkItem tmp;
//   {
//     std::lock_guard<std::mutex> g(mu);
//     tmp = queue.front();
//     queue.pop_front();
//   }
//   process(tmp);  // 锁外做

// =============================================================
// 陷阱 4: 忘记 join / detach
// =============================================================
//
// 反例:
//   void start() {
//     std::thread t([]{ /* 干 10 秒 */ });
//     // 函数返回, t 析构 → std::terminate
//   }
//
// 修复: 用 RAII wrapper
struct ThreadJoiner {
  std::thread t;
  ~ThreadJoiner() {
    if (t.joinable()) t.join();
  }
};

void demo_thread_joiner() {
  cout << "[陷阱 4] ThreadJoiner 演示\n";
  ThreadJoiner j;
  j.t = std::thread([] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cout << "  background done\n";
  });
  // j 析构时自动 join
}

// =============================================================
// 陷阱 5: vector 多线程 push_back — UB
// =============================================================
//
// std::vector::push_back 在容量不够时会 reallocate, 多个线程
// 同时 push 会读到 / 写到已释放的内存, 是 data race, UB。
//
// 修复:
//   - 单线程 push 完再交给多线程读
//   - 用 mutex 保护
//   - 用 tbb::concurrent_vector / folly::AtomicLinkedList

// =============================================================
// 陷阱 6: condition_variable 不能复制
// =============================================================
//
// std::thread, std::mutex, std::condition_variable 都不可拷贝,
// 只能移动。放进容器要 std::move, 用 emplace_back。
//
// 反例:
//   std::vector<std::thread> v;
//   std::thread t([]{});
//   v.push_back(t);    // 编译错: thread 不可拷贝
//
// 修复:
//   v.push_back(std::move(t));
//   v.emplace_back([]{});   // 直接在容器里构造, 不需要移动

// =============================================================
// 陷阱 7: 共享指针循环引用 → 内存泄漏
// =============================================================
//
// 两个对象互相 shared_ptr 引用对方, 引用计数永远不为 0,
// 即使外部所有引用都释放了, 对象也不会析构。
//
// 反例:
//   struct A { std::shared_ptr<B> b; };
//   struct B { std::shared_ptr<A> a; };
//   auto pa = std::make_shared<A>();
//   auto pb = std::make_shared<B>();
//   pa->b = pb;  pb->a = pa;  // 循环引用
//
// 修复: 把其中一方改成 std::weak_ptr。

int main() {
  demo_spurious_wakeup();
  demo_thread_joiner();
  cout << "main exit\n";
  return 0;
}