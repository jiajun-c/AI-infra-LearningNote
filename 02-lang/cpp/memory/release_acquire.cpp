#include <atomic>
#include <thread>
#include <cassert>
#include <string>

struct Payload
{
    int a;
    std::string s;
};

Payload g; 

std::atomic<bool>g_ready(false);

void producer() {
    g.a = 42;
    g.s = "hello";
    g_ready.store(true, std::memory_order_release);
}

void consumer() {
    while (!g_ready.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
    assert(g.a == 42);
    assert(g.s == "hello");
}

int main() {
    std::thread p(producer);
    std::thread s(consumer);
    p.join();
    s.join();
    
}
