// 多生产者 - 多消费者：使用 mutex + condition_variable 实现的有界队列
//
// 编译: g++ -std=c++17 -pthread cv_demo.cpp -o cv_demo
// 运行: ./cv_demo

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// 有界阻塞队列：固定容量，满了生产者阻塞，空了消费者阻塞
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    // 生产者调用：把元素放入队列
    void push(T value) {
        std::unique_lock<std::mutex> lock(mtx_);
        // 谓词：队列未满。wait 会原子地 unlock + block，被唤醒后重新 lock 再检查
        not_full_.wait(lock, [this]{ return queue_.size() < capacity_; });

        queue_.push(std::move(value));
        std::cout << "  [produced] queue size = " << queue_.size() << "\n";

        // 通知一个等待的消费者
        not_empty_.notify_one();
    }

    // 消费者调用：从队列取出元素
    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        // 谓词：队列非空
        not_empty_.wait(lock, [this]{ return !queue_.empty(); });

        T value = std::move(queue_.front());
        queue_.pop();
        std::cout << "[consumed] queue size = " << queue_.size() << "\n";

        // 通知一个等待的生产者
        not_full_.notify_one();
        return value;
    }

    void shutdown() {
        { std::lock_guard<std::mutex> lock(mtx_); stopped_ = true; }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable not_full_;   // 队列有空位时唤醒生产者
    std::condition_variable not_empty_;  // 队列有数据时唤醒消费者
    std::queue<T> queue_;
    size_t capacity_;
    bool stopped_ = false;
};

// ================== 演示 ==================

int main() {
    BoundedQueue<int> q(5);   // 容量为 5 的有界队列

    constexpr int N_PRODUCERS = 2;
    constexpr int N_CONSUMERS = 2;
    constexpr int ITEMS_PER_PRODUCER = 5;

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    // 启动消费者：收到哨兵值 -1 时退出
    std::mutex sum_mtx;
    int total_sum = 0;
    for (int i = 0; i < N_CONSUMERS; ++i) {
        consumers.emplace_back([&]{
            while (true) {
                int v = q.pop();
                if (v == -1) break;          // 哨兵：通知该消费者退出
                std::lock_guard<std::mutex> lock(sum_mtx);
                total_sum += v;
            }
        });
    }

    // 启动生产者：每个生产 ITEMS_PER_PRODUCER 个
    for (int i = 0; i < N_PRODUCERS; ++i) {
        producers.emplace_back([&, id = i]{
            for (int j = 0; j < ITEMS_PER_PRODUCER; ++j) {
                int value = id * 100 + j;
                q.push(value);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }

    // 等待所有生产者完成
    for (auto& t : producers) t.join();

    // 这里消费者还在等数据，我们发送"停止哨兵"让它们退出
    // 简单做法：再 push 一些"特殊值" -1，消费者收到就退出
    for (int i = 0; i < N_CONSUMERS; ++i) q.push(-1);
    for (auto& t : consumers) t.join();

    // 期望 sum = 所有生产者产出的真实值的总和
    // (哨兵 -1 被消费者 break 掉，不会累加到 total_sum)
    int expected = 0;
    for (int i = 0; i < N_PRODUCERS; ++i)
        for (int j = 0; j < ITEMS_PER_PRODUCER; ++j)
            expected += i * 100 + j;

    std::cout << "\ntotal_sum = " << total_sum
              << ", expected = " << expected << "\n";

    return (total_sum == expected) ? 0 : 1;
}