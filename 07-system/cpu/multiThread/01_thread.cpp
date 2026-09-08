#include <chrono>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
static void sleep_ms(int ms) { usleep(static_cast<unsigned int>(ms) * 1000); }

void hello(int id) {
    cout << "[t" << id << "] start, tid=" << std::this_thread::get_id() << "\n";
    sleep_ms(50);
    cout << "[t" << id << "] done\n";
}

int main() {
    const int N = 4;
    cout << "[main] tid = " << std::this_thread::get_id() << "\n";
    std::thread t1([], {
        cout << "[t1] hello from";
        sleep_ms(30);
    })
    
    std::thread t2(hello, 2);
    vector<thread> threads;

    threads.reserve(N);
    
    for (int i = 0; i < N; i++)
    {
        /* code */
        threads.emplace_back(hello, i+1);
    }
    for (auto &t: threads) {
        if (t.joinable()) t.join();
    }
}