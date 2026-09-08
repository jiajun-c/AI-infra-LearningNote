// mpmc_queue.hpp
//
// 阻塞式有界 MPMC（多生产者 / 多消费者）队列，带 shutdown。
//
// 适用场景：
//   - 多个生产者线程往队列里塞任务
//   - 多个消费者线程从队列里取任务
//   - 任意时刻可以调用 shutdown() 让所有线程安全退出
//   - 队列里通常放的是「GPU 任务描述符」（host 指针 + stream + meta），
//     不是 raw GPU buffer，避免拷贝
//
// 设计要点：
//   - 用 std::mutex 保护内部双端队列
//   - 两把 condition_variable：not_full_ 给生产者阻塞，not_empty_ 给消费者阻塞
//   - shutdown 是「排水」语义：消费者先把残留任务消费完，再退出
//   - 不支持拷贝和赋值（队列不该被复制）
//
// 复杂度：
//   - push / pop 是 O(1) 在锁内
//   - 但 mutex/condvar 在高并发下竞争激烈，吞吐量上限在 ~10M ops/sec 量级
//     真要再高，换 moodycamel::ConcurrentQueue 这种 lock-free 实现。

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace sys {

template <typename T>
class MPMCQueue {
 public:
  explicit MPMCQueue(std::size_t capacity)
      : capacity_(capacity), shutdown_(false) {}

  MPMCQueue(const MPMCQueue&) = delete;
  MPMCQueue& operator=(const MPMCQueue&) = delete;

  // ---- 生产者侧 -------------------------------------------------------

  // 阻塞 push。返回 true 表示塞进去了；返回 false 表示队列已被 shutdown。
  bool push(T item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_full_.wait(lock, [this] {
      return shutdown_.load(std::memory_order_acquire) ||
             buffer_.size() < capacity_;
    });
    if (shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    buffer_.push_back(std::move(item));
    lock.unlock();
    // notify_one 而不是 notify_all：唤醒一个消费者即可，避免惊群
    not_empty_.notify_one();
    return true;
  }

  // ---- 消费者侧 -------------------------------------------------------

  // 阻塞 pop。
  //   - 队列里还有东西：取出，返回 true
  //   - 队列空但还没 shutdown：阻塞等待
  //   - 队列空且已 shutdown：返回 false（drain 完成）
  bool pop(T& item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [this] {
      return shutdown_.load(std::memory_order_acquire) || !buffer_.empty();
    });
    if (buffer_.empty()) {
      // 走到这里说明 wait 被唤醒的唯一原因是 shutdown
      return false;
    }
    item = std::move(buffer_.front());
    buffer_.pop_front();
    lock.unlock();
    not_full_.notify_one();
    return true;
  }

  // ---- shutdown / 状态查询 --------------------------------------------

  // 唤醒所有阻塞的生产者和消费者。生产者下次检查会立刻返回 false；
  // 消费者会先把残留任务 drain 完，再返回 false。
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      shutdown_.store(true, std::memory_order_release);
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  bool is_shutdown() const {
    return shutdown_.load(std::memory_order_acquire);
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return buffer_.size();
  }

  std::size_t capacity() const { return capacity_; }

 private:
  mutable std::mutex mu_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> buffer_;     // 用 deque 当底层容器，O(1) push_back / pop_front
  const std::size_t capacity_;
  std::atomic<bool> shutdown_;
};

}  // namespace sys