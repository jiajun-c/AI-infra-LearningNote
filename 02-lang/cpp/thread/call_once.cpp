#include <iostream>
#include <mutex>
#include <thread>

std::once_flag flag;

void initalize() {
    std::cout << "execute once\n";
}

void worker() {
    std::call_once(flag, initalize);
}

int main() {
    std::thread t1(worker);
    std::thread t2(worker);
    std::thread t3(worker);
    t1.join();
    t2.join();
    t3.join();
}