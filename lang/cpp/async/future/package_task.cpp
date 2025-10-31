#include <iostream>
#include <future>
#include <functional>
#include <thread>
#include <cmath>

int f(int x, int y) {return std::pow(x, y); }

void task_lambda() {
    std::packaged_task<int(int, int)> task(
        [](int x, int y) {  return std::pow(x, y); }
    );
    std::future<int> result = task.get_future();
    task(2, 3);
    std::cout << "task_lambda: " << result.get() << std::endl;

}

void test_bind() {
    std::packaged_task<int()> task(
        std::bind(f, 2, 3)
    );
    std::future<int> result = task.get_future();
    task();
    std::cout << "test_bind: " << result.get() << std::endl;
}

void task_thread()
{
    std::packaged_task<int(int, int)> task(f);
    std::future<int> result = task.get_future();
 
    std::thread task_td(std::move(task), 2, 3);
    task_td.join();
 
    std::cout << "task_thread:\t" << result.get() << '\n';
}
 
int main() {
    task_lambda();
    test_bind();
    task_thread();
    return 0;
}