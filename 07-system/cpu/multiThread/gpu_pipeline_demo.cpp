// gpu_pipeline_demo.cpp
//
// 一个典型的「CPU 多线程 + GPU」流水线，演示 MPMCQueue 的用法：
//
//   生产者线程 (n个)                     消费者线程 (m个)
//   ┌──────────────┐                   ┌──────────────┐
//   │ 申请 host buf│                   │  launch kernel│
//   │ cudaMemcpy    │  ──── push ───> │  cudaStreamSync
//   │  (async)     │                   │  release buf  │
//   └──────────────┘                   └──────────────┘
//            │                                  │
//            └───────── MPMCQueue<Work> ────────┘
//
// 这个 demo 只在 CPU 上跑，不真的调用 CUDA，所以编译只需要支持 C++17：
//   g++ -std=c++17 -O2 gpu_pipeline_demo.cpp -lpthread -o gpu_pipeline_demo
//   ./gpu_pipeline_demo

#include "mpmc_queue.hpp"

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

// 模拟一个 "GPU 任务" 的描述符。
// 真实场景里通常还会有 cudaStream_t、device pointer 之类的字段，
// 但 demo 里我们只放 host 端的元信息。
struct GpuWork {
  std::shared_ptr<float> host_buf;   // 模拟一段输入数据
  int batch_size;
  int seq_id;
};

static void sleep_ms(int ms) {
  usleep(static_cast<unsigned int>(ms) * 1000);
}

int main() {
  // 有界队列：capacity 不宜太大，否则相当于"无界"了
  sys::MPMCQueue<GpuWork> q(4);

  constexpr int kNumProducers = 3;
  constexpr int kNumConsumers = 2;
  constexpr int kItemsPerProducer = 20;

  std::atomic<int> produced{0};
  std::atomic<int> consumed{0};

  // ---- 生产者线程 ---------------------------------------------------
  std::vector<std::thread> producers;
  for (int p = 0; p < kNumProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kItemsPerProducer; ++i) {
        GpuWork w;
        w.host_buf = std::make_shared<float>(1024);
        w.batch_size = 8;
        w.seq_id = p * 1000 + i;

        // 模拟 H2D + 一点点 CPU 计算
        sleep_ms(2);

        if (!q.push(std::move(w))) {
          // 队列被 shutdown 了，放弃剩下的工作
          std::cout << "[producer " << p << "] saw shutdown, exit\n";
          return;
        }
        produced.fetch_add(1);
      }
    });
  }

  // ---- 消费者线程 ---------------------------------------------------
  std::vector<std::thread> consumers;
  for (int c = 0; c < kNumConsumers; ++c) {
    consumers.emplace_back([&, c] {
      while (true) {
        GpuWork w;
        if (!q.pop(w)) {
          // drain 完了 + shutdown，退出
          std::cout << "[consumer " << c << "] drained, exit\n";
          return;
        }

        // 模拟 GPU kernel 启动 + 同步
        sleep_ms(5);

        consumed.fetch_add(1);
      }
    });
  }

  // ---- 主线程：等所有生产者做完，然后 shutdown -----------------------
  for (auto& t : producers) t.join();
  std::cout << "[main] all producers done, produced=" << produced.load() << "\n";

  q.shutdown();   // 通知消费者：drain 完就退
  for (auto& t : consumers) t.join();

  std::cout << "[main] consumed=" << consumed.load()
            << " (expected " << kNumProducers * kItemsPerProducer << ")\n";

  return 0;
}