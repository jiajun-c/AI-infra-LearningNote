#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <immintrin.h>

using namespace std;

struct MCSNode {
    std::atomic<MCSNode*> next{nullptr};
    std::atomic<bool>     locked{false};
};

class MCSLock {
    std::atomic<MCSNode*> tail_{nullptr};
public:
    void lock(MCSNode* node) {
        node->next.store(nullptr);
        node->locked.store(false);
        MCSNode* prev = tail_.exchange(node, std::memory_order_acq_rel);
        if (prev == nullptr) return;            // 第一个拿锁
        prev->next.store(node, std::memory_order_release);
        // 进入临界区
        while (!node->locked.load(std::memory_order_acquire)) _mm_pause();
    }
    void unlock(MCSNode* node) {
        if (node->next.load(std::memory_order_acquire) == nullptr) {
            // 没有后继，尝试清空 tail
            MCSNode* expected = node;
            if (tail_.compare_exchange_strong(expected, nullptr,
                    std::memory_order_release,
                    std::memory_order_relaxed)) return;
            // 有后继在等，等它把 next 挂上
            while (node->next.load(std::memory_order_acquire) == nullptr) _mm_pause();
        }
        // 离开临界区
        node->next.load(std::memory_order_acquire)->locked.store(true, std::memory_order_release);
    }
};


int main() {
    constexpr int N = 4;          // 线程数
    constexpr int M = 100000;     // 每线程加锁次数

    MCSLock      lock;
    long long    counter = 0;     // 临界区内的共享变量

    auto worker = [&]() {
        // MCS 锁要求每个线程持有自己的 node，thread_local 天然满足这一点
        thread_local MCSNode node;
        for (int i = 0; i < M; ++i) {
            lock.lock(&node);
            ++counter;
            lock.unlock(&node);
        }
    };

    vector<thread> threads;
    for (int i = 0; i < N; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    cout << "expect: " << (long long)N * M << '\n';
    cout << "actual: " << counter << '\n';
    return counter == (long long)N * M ? 0 : 1;
}
