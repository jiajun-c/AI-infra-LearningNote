// 06_thread_pool.cpp
//
// Lesson 06: 简单线程池
//   - 一组 worker 线程 + 任务队列
//   - submit 提交 std::function<void()>, 返回 std::future<R>
//   - shutdown 排空队列后退出
//   - 综合运用 mutex + condvar + future
//
// 编译: g++ -std=c++17 -O2 -pthread 06_thread_pool.cpp -o /tmp/06
// 运行: /tmp/06

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t n_workers) {
    workers_.reserve(n_workers);
    for (std::size_t i = 0; i < n_workers; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // 提交一个有返回值的任务
  template <typename F>
  auto submit(F&& f) -> std::future<decltype(f())> {
    using R = decltype(f());
    // 用 shared_ptr 包装, 让 task 可拷贝 (packaged_task 本身不可拷贝)
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut = task->get_future();

    {
      std::lock_guard<std::mutex> g(mu_);
      tasks_.emplace_back([task]() { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  // shutdown: 排空队列后 worker 退出
  void shutdown() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;  // 排干后才退
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();  // 锁外执行, 不阻塞其他 worker
    }
  }

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

int main() {
  ThreadPool pool(4);  // 4 个 worker

  // 提交 8 个任务
  std::vector<std::future<int>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.push_back(pool.submit([i] {
      // 模拟耗时
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return i * i;
    }));
  }

  // 取结果 (会阻塞, 但 4 worker 并行)
  for (int i = 0; i < 8; ++i) {
    std::cout << "task " << i << " -> " << futures[i].get() << "\n";
  }

  pool.shutdown();
  std::cout << "all done\n";
  return 0;
}