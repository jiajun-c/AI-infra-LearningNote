# 进程

进程一般由三个部分组成
- 进程控制块 PCB，包含进程标识符PID，进程当前状态，程序和数据地址，进程优先级，CPU现场保护区(用于进程切换)，占有的资源清单等
- 程序段
- 数据段

## 进程通信的方式
- 管道通信
- 系统IPC（消息队列，信号量，信号，共享内存）
- 套接字socket

## 进程调度时机

- 当前运行的进程进行结束
- 当前运行的进程由于某种原因阻塞
- 执行完系统调用等系统程序后返回用户进程
- 在使用抢占调度的系统中，具有更高优先级的进程就绪时
- 分时系统中，分给当前进程的时间片用完

# 线程

线程完全由操作系统内核一手包办，线程分为内核级线程和用户级线程。

内核线程由os内核管理，操作系统知情，用户级线程由用户库管理，操作系统不知情

线程和进程的对应关系是1:N，一个正常的组成是 进程 = 资源（内存/文件/设备） + 一堆线程（至少一个），如在多线程程序中，该程序的代码是共享的，同时分配在堆上动态申请的内存也是共享的，而栈，寄存器等则是线程所私有的

如下所示，我们可以看到这些线程的pid都是相同的，但是他们的tid是不同的，同时运行在不同的GPU上

```cpp
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
```

# 协程

协程本身是一个轻量级的线程，其实现由语言实现，例如go中的协程机制

协程本身也可以分为有栈携程和无栈携程

