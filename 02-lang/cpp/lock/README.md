# C++ Lock

## 1. 第一层

### 1.0 RAII 封装

假设不使用RAII进行封装，将会导致线程在中途退出的时候锁得不到释放，从而导致程序挂起，如下所示

```cpp
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

void err() {
    throw std::runtime_error("boom");
}

std::mutex m;
int sum = 0;
void func(int i) {
    if (i==2) {
        std::this_thread::sleep_for(std::chrono::seconds(1));                                   
    }
    m.lock();
    if(i==1) err();
    sum++;
    m.unlock();
}

int main() {
    
    std::thread t1([]{
        try { func(1); }                                                                   
        catch (const std::exception& e) { std::cout << "t1 捕获: " << e.what() << '\n'; }
    });   
    std::thread t2(func, 2);
    t1.join();
    t2.join();
    printf("%d\n", sum);
}
```

RAII的包装器有四个

- std::lock_guard:一把锁，默认
- std::unique_lock: 可以手动解锁和延迟加锁
- std::scoped_lock：可以锁多把锁
- std::shared_lock: 可以锁一把锁，配合shared_mutex来读锁

```cpp
#include <iostream>
#include <mutex>
#include <vector>
#include <thread>
#include <string>

using namespace std;

struct Account
{
    /* data */
    std::mutex m;
    long balance;
};


void transfer(Account& from, Account& to, long amount) {                                 
    std::scoped_lock lock(from.m, to.m);   // 不是 from.m.lock() 再 to.m.lock()          
    if (from.balance < amount) return;                                                   
    from.balance -= amount;                                                              
    to.balance += amount;                                                                
}

int main() {
    Account a{{}, 1000};
    Account b{{}, 1000};
    std::thread t1([&]{
        transfer(a, b, 100);
    });
    std::thread t2([&]{
        transfer(b, a, 150);
    });
    t1.join();
    t2.join();
    printf("%d %d\n",a.balance, b.balance);
}
```
  
### 1.1 std::mutex

std::mutex有三个接口

```cpp
try_lock(); // 尝试lock，不行就返回，不阻塞
lock();
unlock();
```

### 1.2 std::shared_mutex

这个主要是用于让读线程可以并行，但是写线程只能有一个在写，但是读线程可以并行，同时读写线程之间也是互斥的

```cpp
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
```

### 1.3 