#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

// 全局共享的 vector
std::vector<int> g_vec;

void worker_func() {
    // 每个线程疯狂插入 10000 个数据
    // 这会触发多次 vector 的扩容 (Reallocation)
    for (int i = 0; i < 10000; ++i) {
        // 🛑 这里没有加锁！
        // 多个线程同时修改 size、capacity 和内存指针
        g_vec.push_back(i); 
    }
}

int main() {
    std::cout << "开始多线程 push_back 测试..." << std::endl;

    // 启动 5 个线程，增加冲突概率
    std::thread t1(worker_func);
    std::thread t2(worker_func);
    std::thread t3(worker_func);
    std::thread t4(worker_func);
    std::thread t5(worker_func);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    std::cout << "最终大小: " << g_vec.size() << std::endl;
    return 0;
}