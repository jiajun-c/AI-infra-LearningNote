# CPU 伪共享

CPU伪共享指的是多核CPU缓存一致性所带来的一个性能问题

在CPU中缓存不是以字节为单位进行管理的，而是以缓存行（cache line）为单位，典型的缓存行大小是64字节

当不同的核去同时修改了一个cacheline，将会导致其在内存中反复失效

如下所示使用alignas(cache_line_size)将8字节的强制对齐到64字节从而消除了伪共享


```cpp 
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// 使用 C++17 特性获取缓存行大小，如果编译器不支持则默认 64
#ifdef __cpp_lib_hardware_interference_size
    #include <new>
    constexpr size_t cache_line_size = std::hardware_destructive_interference_size;
#else
    constexpr size_t cache_line_size = 64;
#endif

// 场景 1：会产生伪共享的结构
struct FalseSharingData {
    volatile long val1;
    volatile long val2;
};

// 场景 2：对齐后的结构（消除伪共享）
struct AlignedData {
    alignas(cache_line_size) volatile long val1;
    alignas(cache_line_size) volatile long val2;
};

const int ITERATIONS = 100000000;

void worker_func(volatile long* val) {
    for (int i = 0; i < ITERATIONS; ++i) {
        (*val)++;
    }
}

int main() {
    std::cout << "Detected Cache Line Size: " << cache_line_size << " bytes\n\n";
    
    // 测试 1：伪共享
    FalseSharingData fs_data = {0, 0};
    auto start1 = std::chrono::high_resolution_clock::now();
    std::thread t1(worker_func, &fs_data.val1);
    std::thread t2(worker_func, &fs_data.val2);
    t1.join();
    t2.join();
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    std::cout << "Time with False Sharing: " << elapsed1.count() << " ms\n";
    // 测试 2：消除伪共享
    AlignedData al_data = {0, 0};
    auto start2 = std::chrono::high_resolution_clock::now();
    std::thread t3(worker_func, &al_data.val1);
    std::thread t4(worker_func, &al_data.val2);
    t3.join();
    t4.join();
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    std::cout << "Time with Padding (Aligned): " << elapsed2.count() << " ms\n";
    return 0;
    
}
```

