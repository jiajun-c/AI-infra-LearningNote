#include <iostream>
#include <future>
#include <thread>

using namespace std;

int main() {
    std::packaged_task<int()>task([]{return 7;});
    std::future<int>f1 = task.get_future();
    std::thread t(std::move(task));
}