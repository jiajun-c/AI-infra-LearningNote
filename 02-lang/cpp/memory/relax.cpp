#include <atomic>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>

std::atomic<long> g_counter{0};

void worker(int n) {
    for (int i = 0; i < n; i++) {
        g_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

int main() {
    std::vector<std::thread>ts;
    for (int i = 0; i < 8; i++) {
        ts.emplace_back(worker, 1000);
    }

    for (auto &t: ts) {
        t.join();
    }

    printf("%d\n", g_counter.load(std::memory_order_relaxed));
}