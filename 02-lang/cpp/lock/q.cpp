#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

template <
    typename T,
    std::size_t Capacity,
    std::size_t ConsumerCount
>
class BroadcastRing {
    static_assert(Capacity > 0 &&
                  (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");

    static_assert(ConsumerCount > 0,
                  "At least one consumer is required");

    static_assert(std::is_default_constructible_v<T>,
                  "T must be default constructible");

    static_assert(std::is_nothrow_copy_assignable_v<T>,
                  "T must support noexcept copy assignment");

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "Requires lock-free 64-bit atomics");

    using Sequence = std::uint64_t;

    // 64 字节用于常见平台上的缓存行隔离
    struct alignas(64) ConsumerState {
        // 已完成读取的元素数量，生产者会读取它
        std::atomic<Sequence> consumed{0};

        // 只由对应消费者线程访问
        Sequence next = 0;
        Sequence cachedPublished = 0;
    };

    struct alignas(64) ProducerState {
        // 只由生产者线程访问
        Sequence next = 0;
        Sequence cachedMinConsumed = 0;
    };

public:
    BroadcastRing() = default;

    BroadcastRing(const BroadcastRing&) = delete;
    BroadcastRing& operator=(const BroadcastRing&) = delete;

    // 只能由唯一的生产者线程调用
    bool try_push(const T& value) noexcept {
        const Sequence pos = producer_.next;

        // 缓存显示可能已满，才重新读取消费者进度
        if (pos - producer_.cachedMinConsumed >= Capacity) {
            Sequence minimum = pos;

            for (std::size_t i = 0; i < ConsumerCount; ++i) {
                const Sequence consumed =
                    consumers_[i].consumed.load(
                        std::memory_order_acquire);

                if (consumed < minimum) {
                    minimum = consumed;
                }
            }

            producer_.cachedMinConsumed = minimum;

            if (pos - minimum >= Capacity) {
                return false;
            }
        }

        buffer_[pos & (Capacity - 1)] = value;
        producer_.next = pos + 1;

        // 先写数据，再发布：消费者 acquire 后可读取数据
        published_.store(pos + 1, std::memory_order_release);
        return true;
    }

    // consumerId 必须小于 ConsumerCount
    // 同一个 consumerId 只能由一个消费者线程调用
    bool try_pop(std::size_t consumerId, T& value) noexcept {
        ConsumerState& state = consumers_[consumerId];
        const Sequence pos = state.next;

        // 已知可读范围用完后，才重新读取发布位置
        if (pos == state.cachedPublished) {
            state.cachedPublished =
                published_.load(std::memory_order_acquire);

            if (pos == state.cachedPublished) {
                return false;
            }
        }

        // 每个消费者复制自己的数据，不能 move 共享槽位
        value = buffer_[pos & (Capacity - 1)];

        state.next = pos + 1;

        // 复制完成后才能允许生产者复用槽位
        state.consumed.store(
            pos + 1, std::memory_order_release);

        return true;
    }

private:
    alignas(64) std::array<T, Capacity> buffer_{};

    alignas(64) std::atomic<Sequence> published_{0};

    ProducerState producer_;

    std::array<ConsumerState, ConsumerCount> consumers_{};
};