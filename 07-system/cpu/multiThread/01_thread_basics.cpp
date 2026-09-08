// 01_thread_basics.cpp
//
// Lesson 01: std::thread 基础
//   - 启动线程
//   - join vs detach
//   - 参数按值传递（即使是指针）
//   - this_thread 工具函数
//
// 编译: g++ -std=c++17 -O2 -pthread 01_thread_basics.cpp -o /tmp/01
// 运行: /tmp/01

#include <chrono>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace std;

static void sleep_ms(int ms) { usleep(static_cast<unsigned int>(ms) * 1000); }

// 普通函数也可以作为线程入口
void hello(int id) {
  cout << "[t" << id << "] start, tid=" << std::this_thread::get_id() << "\n";
  sleep_ms(50);
  cout << "[t" << id << "] done\n";
}

int main() {
  cout << "[main] tid=" << std::this_thread::get_id() << "\n";

  // ---- 1. 启动一个线程，传 lambda ----
  std::thread t1([] {
    cout << "[t1] hello from lambda\n";
    sleep_ms(30);
  });

  // ---- 2. 启动一个线程，传函数 + 参数 ----
  std::thread t2(hello, 2);

  // ---- 3. 参数按值拷贝 ----
  // 下面这行哪怕 x 是引用类型，传进去的也是值的拷贝。
  // 想传引用要用 std::ref。
  int x = 42;
  std::thread t3([x] {
    cout << "[t3] captured x = " << x << " (按值拷贝)\n";
  });

  // ---- 4. join 阻塞等到线程结束 ----
  t1.join();
  t2.join();
  t3.join();

  cout << "[main] all joined\n";

  // ---- 5. detach 让线程在后台独立运行 ----
  // 注意: detach 之后不能再 join; 也不能再访问主线程栈上的局部变量。
  std::thread t4([] {
    sleep_ms(20);
    cout << "[t4] running detached\n";
  });
  t4.detach();

  // 主线程等 t4 一会儿，否则主线程退出时 t4 也跟着死。
  sleep_ms(50);
  cout << "[main] exit\n";

  // 关键规则: std::thread 对象销毁前，必须 join 或 detach，否则 std::terminate。
  // 一个常用 RAII 包装:
  //   struct ThreadJoiner { std::thread t;
  //     ~ThreadJoiner() { if (t.joinable()) t.join(); } };
  return 0;
}