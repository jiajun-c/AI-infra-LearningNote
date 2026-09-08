// 03_condition_variable.cpp
//
// Lesson 03: std::condition_variable
//   - condvar 配合 unique_lock 使用
//   - wait 必须带 predicate (避免假唤醒)
//   - notify_one vs notify_all
//   - notify 之前先 unlock
//
// 编译: g++ -std=c++17 -O2 -pthread 03_condition_variable.cpp -o /tmp/03
// 运行: /tmp/03

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

// ---------- 例 1: 一生产者一消费者 ----------

static std::mutex mu;
static std::condition_variable cv;
static bool data_ready = false;
static int payload = 0;

void producer() {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 模拟准备
  {
    std::lock_guard<std::mutex> g(mu);
    payload = 42;
    data_ready = true;
    // 锁还在手里就 notify, 被唤醒的消费者会被 mutex 挡住一次,
    // 但标准允许这种写法 (虚唤醒 + predicate 重检能兜住)。
  }
  cv.notify_one();
}

void consumer() {
  std::unique_lock<std::mutex> lk(mu);
  // wait 的 predicate 版本是 while(!pred) wait(lk) 的语法糖,
  // 推荐永远用这个版本, 防止假唤醒和 lost wakeup。
  cv.wait(lk, [] { return data_ready; });
  cout << "[consumer] got payload = " << payload << "\n";
}

void demo_one_to_one() {
  cout << "=== demo 1: 1 producer + 1 consumer ===\n";
  data_ready = false;
  payload = 0;
  std::thread t_producer(producer);
  std::thread t_consumer(consumer);
  t_producer.join();
  t_consumer.join();
  cout << "\n";
}

// ---------- 例 2: 多消费者, notify_all ----------
//
// 一个数据可能要被多个消费者瓜分, 用 notify_all 更合适。

static bool shutdown = false;

void worker(int id) {
  std::unique_lock<std::mutex> lk(mu);
  while (!shutdown) {  // 外层 while 是"再等下一轮"
    cv.wait(lk, [] { return data_ready || shutdown; });
    if (shutdown) break;
    cout << "[worker " << id << "] consumes " << payload << "\n";
    data_ready = false;  // 消费完清空, 通知 producer 可以继续
    lk.unlock();
    cv.notify_all();   // 通知 producer
    lk.lock();
  }
}

void producer_two() {
  std::unique_lock<std::mutex> lk(mu);
  for (int i = 0; i < 5; ++i) {
    cv.wait(lk, [] { return !data_ready; });  // 等消费者消费完
    payload = i * 10;
    data_ready = true;
    cout << "[producer] put " << payload << "\n";
    lk.unlock();
    cv.notify_all();   // 唤醒所有 worker
    lk.lock();
  }
  // 通知大家收工
  shutdown = true;
  lk.unlock();
  cv.notify_all();
}

void demo_multi() {
  cout << "=== demo 2: 1 producer + 3 consumers ===\n";
  data_ready = false;
  shutdown = false;
  payload = 0;

  std::thread w1(worker, 1);
  std::thread w2(worker, 2);
  std::thread w3(worker, 3);
  std::thread p(producer_two);

  p.join();
  w1.join();
  w2.join();
  w3.join();
}

int main() {
  demo_one_to_one();
  demo_multi();
  return 0;
}