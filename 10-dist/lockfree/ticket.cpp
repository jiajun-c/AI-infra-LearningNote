#include <iostream>
#include <vector>
#include <atomic>

using namespace std;

class TicketLock {
    std::atomic<uint64_t> next_{0}, now_{0};
public:
    void lock() {
        uint64_t t = next_.fetch_add(1, std::memory_order_relaxed);
        while (now_.load(std::memory_order_acquire) != t) _mm_pause();
    }
    void unlock() {
        now_.fetch_add(1, std::memory_order_release);
    }
};


int main() {

}