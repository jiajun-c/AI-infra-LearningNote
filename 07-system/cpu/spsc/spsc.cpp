#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <cassert>
#include <immintrin.h>
// 假设缓存行大小为 64 字节
constexpr size_t CACHE_LINE_SIZE = 64;

template <typename T, size_t Capacity>
class SPSCQueue {
private:
    // TODO 1: 内存布局优化
    // 回顾我们刚刚讲过的伪共享知识。
    // 生产者频繁修改 tail，消费者频繁修改 head。
    // 你应该如何使用 alignas 来优化这些成员变量的位置？
    
    T buffer[Capacity];
    std::atomic<size_t> head{0}; // 消费者读取位置 (Read Index)
    std::atomic<size_t> tail{0}; // 生产者写入位置 (Write Index)

public:
    // 尝试写入数据。如果队列满，返回 false (非阻塞设计)
    bool push(const T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;
        
        // [优化 3]：读取消费者的 head，必须确保看到最新的值，用 acquire
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; // 队列满
        }
        
        // 先写数据
        buffer[current_tail] = item;
        
        // [优化 4]：把 tail 的更新发布出去，并保证前面的 buffer 写入一定先发生，用 release
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    // 尝试读取数据。如果队列空，返回 false (非阻塞设计)
    bool pop(T& item) {
size_t current_head = head.load(std::memory_order_relaxed);
        
        // [优化 3]：读取生产者的 tail，看是否有新数据，用 acquire
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // 队列空
        }
        
        // [修复致命 Bug]：把数据实打实地读出来！
        item = buffer[current_head]; 
        
        size_t next_head = (current_head + 1) % Capacity;
        
        // [优化 4]：更新 head 告诉生产者有空位了，用 release
        head.store(next_head, std::memory_order_release);
        return true;
    }
};

// ==========================================
// 评测框架 (无需修改)
// ==========================================
const int NUM_ITEMS = 10000000; // 测试 1000 万次读写

void producer_thread(SPSCQueue<int, 1024>& queue) {
    for (int i = 0; i < NUM_ITEMS; ++i) {
        // 只要队列满，就一直重试 (Busy Spin)
        while (!queue.push(i)) {
            // 在实际 HFT 中，这里可能会用到 _mm_pause()
            _mm_pause();
        }
    }
}

void consumer_thread(SPSCQueue<int, 1024>& queue) {
    int item;
    for (int i = 0; i < NUM_ITEMS; ++i) {
        // 只要队列空，就一直重试 (Busy Spin)
        while (!queue.pop(item)) {
            // spin
        }
        // 验证数据的一致性
        if (item != i) {
            std::cerr << "Data mismatch! Expected " << i << ", got " << item << "\n";
            std::exit(1);
        }
    }
}

int main() {
    SPSCQueue<int, 1024> queue;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::thread prod(producer_thread, std::ref(queue));
    std::thread cons(consumer_thread, std::ref(queue));

    prod.join();
    cons.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    std::cout << "Successfully passed all data.\n";
    std::cout << "Time taken for " << NUM_ITEMS << " items: " << elapsed.count() << " ms\n";
    std::cout << "Throughput: " << (NUM_ITEMS / (elapsed.count() / 1000.0)) / 1000000.0 << " Million Ops/sec\n";

    return 0;
}