#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>       // 包含 sched_getcpu()

std::mutex print_mutex;

void worker_function(int worker_num) {
    // 这是一个忙等待循环，确保占住 CPU，防止瞬间执行完
    // 只有占住 CPU，才能更有机会观察到它们分布在不同核上
    long long counter = 0;
    while(counter < 100000000) { counter++; }

    std::lock_guard<std::mutex> lock(print_mutex);
    
    // 获取当前线程运行在哪个 CPU 核心上
    int cpu_id = sched_getcpu();
    
    std::cout << "[子线程 " << worker_num << "] "
              << "PID: " << getpid() 
              << " | TID: " << syscall(SYS_gettid)
              << " | \033[1;31m运行在 CPU: " << cpu_id << "\033[0m" // 红色高亮 CPU 号
              << std::endl;
}

int main() {
    int num_cores = std::min(4, int(std::thread::hardware_concurrency()));
    std::cout << "当前电脑拥有逻辑核心数: " << num_cores << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::vector<std::thread> threads;
    // 启动和核心数一样多的线程，大概率会占满不同核心
    for (int i = 0; i < num_cores; ++i) {
        threads.emplace_back(worker_function, i);
    }

    for (auto& t : threads) {
        t.join();
    }
    return 0;
}