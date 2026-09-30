#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

template<typename T, std::size_t Capacity>
class SpscRingQueue {
    static_assert(Capacity >= 2);
public:
    SpscRingQueue() = default;
    SpscRingQueue(const SpscRingQueue&) = delete;
    SpscRingQueue& operator = (const SpscRingQueue&) = delete;

    template<typename U>
    bool try_push(U&& value) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t next_tail = next(tail);
        if (next_tail == head_.load(std::memory_order_acquire)) return false;
        buffer_[tail] = std::forward<U>(value);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool try_pop(T& value) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) return false;
        value = std::move(buffer_[head]);
        head_.store(next(head), std::memory_order_release);
        return true;
    }
private:
    static constexpr std::size_t next(std::size_t index) noexcept {
        return (index + 1)%Capacity;
    }
    std::array<T, Capacity>buffer_{};
    alignas(64) std::atomic<std::size_t>head_{0};
    alignas(64) std::atomic<std::size_t>tail_{0};
};

#include <iostream>
#include <thread>

int main() {
    constexpr int element_count = 10;
    SpscRingQueue<int, 4> queue;

    std::thread producer([&] {
        for (int i = 0; i < element_count; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&] {
        for (int expected = 0; expected < element_count; ++expected) {
            int value;

            while (!queue.try_pop(value)) {
                std::this_thread::yield();
            }

            std::cout << "pop value = " << value
                      << ", expected = " << expected
                      << (value == expected ? " [OK]" : " [ERROR]")
                      << '\n';
        }
    });

    producer.join();
    consumer.join();
}
