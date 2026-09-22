#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <iostream>
using namespace std;


std::mutex m;
int sum = 0;
void func() {
    m.lock();
    sum++;
    m.unlock();
}

int main() {
    std::thread t1(func);
    std::thread t2(func);
    t1.join();
    t2.join();
    printf("%d\n", sum);
}