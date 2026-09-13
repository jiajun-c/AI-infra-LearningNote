#include <iostream>
#include <vector>
#include <atomic>

using namespace std;

class CASLock {
    std::atomic<bool> locked{false};
public:
    void lock() {
        while (locked.exchange(true, std::memory_order_acquire)) {
            // 自旋 / 退避
        }
    }
    void unlock() {
        locked.store(false, std::memory_order_release);
    }
};