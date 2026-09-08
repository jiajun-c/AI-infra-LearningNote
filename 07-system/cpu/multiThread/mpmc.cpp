#include <iostream>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>

template <typename T>
class MPMCQueue {
    public:
        explicit MPMQueue(std::size_t capacity)
            :capacity_(capacity), shutdown_(false) {}
        MPMCQueue(const MPMCQueue&) = delete;
        MPMCQueue& operator=(const MPMCQueue&) = delete;

        bool push(T item) {
            std::unique_lock<std::mutex>lock(mu_);
            not_full_.wait(lock, [this]{
                return shutdown_.load(std::memory_order_acquire) || 
                        buffer_.size() < capacity_;
            });
            if (shutdown_.load(std::memory_order_acquire)) {
                return false;
            }
            buffer_.push_back(std::move(item));
            lock.unlock();
            not_empty_.notify_one();
            return true;
        }

        bool pop(T& item) {
            std::unique_lock<std::mutex> lock(mu_);
            not_empty_.wait(lock, [this]{
                return shutdown_.load(std::memory_order_acquire) || !buffer_.empty();
            })
            if (buffer_.empty()) {
                return false;
            }

            item = std::move(buffer_.front());
            buffer_.pop_front();
            lock.unlock();
            not_full_.notify_one();
            return true;
        }
    private:
        mutable std::mutex mu_;
        std::condition_variable not_empty_;
        std::condition_variable not_full_;
        std::deque<T> buffer_;
        const std::size_t capacity;
        std::atomic<bool>shutdown_;

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
  