#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <iostream>
#include <shared_mutex>
#include <stdexcept>
#include <time.h>
#include <chrono>
using namespace std;


std::shared_mutex m;
int sum = 0;
void reader() {
    std::shared_lock lock(m);
    printf("sum: %d\n", sum);
}

void writer() {
    std::unique_lock lock(m);
    sum++;
}

int main() {
    
    std::thread r1(reader), r2(reader);
    std::thread t1(writer), t2(writer);
    t1.join();
    t2.join();
    r1.join();
    r2.join();

    printf("%d\n", sum);
}