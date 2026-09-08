// 05_future_and_promise.cpp
//
// Lesson 05: std::future / std::promise / std::async
//   - future 是"一次性同步"语义: 子线程把结果交给主线程
//   - promise 是写端, future 是读端
//   - async 是更短的写法
//   - shared_future 可以被多个线程 wait
//
// 编译: g++ -std=c++17 -O2 -pthread 05_future_and_promise.cpp -o /tmp/05
// 运行: /tmp/05

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

using namespace std;

// ---------- 例 1: 手动 promise + future ----------

void producer(std::promise<int> p) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  p.set_value(42);   // 把值传出去, 唤醒对应的 future
}

void demo_promise_future() {
  cout << "=== demo 1: promise + future ===\n";
  std::promise<int> p;
  std::future<int> f = p.get_future();

  std::thread t(producer, std::move(p));  // promise 只能 move
  cout << "[main] waiting...\n";
  int v = f.get();   // 阻塞等到 set_value
  cout << "[main] got " << v << "\n\n";
  t.join();
}

// ---------- 例 2: std::async 一步到位 ----------

int compute(int x) {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  return x * x;
}

void demo_async() {
  cout << "=== demo 2: std::async ===\n";
  // std::launch::async 强制新线程; 不指定的话实现可能选 deferred (惰性求值)
  std::future<int> f1 = std::async(std::launch::async, compute, 7);
  std::future<int> f2 = std::async(std::launch::async, compute, 9);

  cout << "[main] f1 = " << f1.get() << "\n";
  cout << "[main] f2 = " << f2.get() << "\n\n";
}

// ---------- 例 3: shared_future 多线程等同一个值 ----------

void worker(std::shared_future<int> sf, int id) {
  int v = sf.get();   // shared_future 可以被多个线程 get, 各自拿到一份拷贝
  cout << "[worker " << id << "] got " << v << "\n";
}

void demo_shared_future() {
  cout << "=== demo 3: shared_future ===\n";
  std::promise<int> p;
  std::shared_future<int> sf = p.get_future().share();   // 一次性 share

  std::thread t1(worker, sf, 1);
  std::thread t2(worker, sf, 2);
  std::thread t3(worker, sf, 3);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  p.set_value(99);

  t1.join();
  t2.join();
  t3.join();
  cout << "\n";
}

// ---------- 例 4: future + 超时 ----------
//
// wait_for 返回状态: ready / timeout / deferred。

void demo_wait_for() {
  cout << "=== demo 4: wait_for 超时 ===\n";
  std::future<int> f = std::async(std::launch::async, [] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    return 1;
  });

  auto status = f.wait_for(std::chrono::milliseconds(50));
  if (status == std::future_status::timeout) {
    cout << "[main] 50ms 内还没好, 不等了, 让它在后台跑\n";
  }
  // 不 get 也不 wait, 析构时 future 会阻塞等子线程 (有点坑)
  cout << "[main] exit, future 析构会 block 直至后台线程结束\n";
}

int main() {
  demo_promise_future();
  demo_async();
  demo_shared_future();
  demo_wait_for();
  return 0;
}