# C++多线程

## 1.基础用法

创建一个线程，并且传入一个lambda函数供线程调用。

```cpp
    std::thread t1([], {
        cout << "[t1] hello from";
        sleep_ms(30);
    })
```

有两个重要的函数

- joinable：检查线程是否还或者
- join：等待线程运行结束，然后回收线程

```cpp
    for (auto &t: threads) {
        if (t.joinable()) t.join();
    }
```


传入多线程调用函数的参数，如下所示，第一个参数是调用的函数，后续的参数是函数的参数

```cpp
void hello(int id) {
  cout << "[t" << id << "] start, tid=" << std::this_thread::get_id() << "\n";
  sleep_ms(50);
  cout << "[t" << id << "] done\n";
}

std::thread t2(hello, 2);

```

