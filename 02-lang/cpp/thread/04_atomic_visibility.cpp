// 04_atomic_visibility.cpp
//
// 主题: 多线程下的「可见性」(visibility) 与 std::atomic 内存序
//
// 四个 demo 对比:
//   1) 普通变量 — data race, UB, x86 上常常"看起来"对, 但永远不该这么写
//   2) std::atomic 默认序 (seq_cst) — 最安全的全局顺序一致
//   3) std::atomic acquire / release — 经典 synchronizes-with 配对, 正确且高效
//   4) std::atomic relaxed — flag 是 relaxed 时, 不能同步 payload 写入, payload 可能旧
//
// 编译: g++ -std=c++17 -O2 -pthread 04_atomic_visibility.cpp -o /tmp/04_atomic_vis
// 运行: /tmp/04_atomic_vis

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std;
using namespace std::chrono_literals;

// ============================================================================
// Demo 1: 普通变量, 没有同步
// ============================================================================
//
// 经典错法: "先写 data, 再写 flag; 等 flag 再读 data"。
// 这在 C++ 内存模型里是 UB, 因为 data / flag 两个变量之间没有 happens-before 关系:
//   - 编译器有权把 flag 的写提前
//   - CPU 弱序架构 (ARM / RISC-V) 硬件本身也会重排
//   - x86 (TSO) 上大多数时候能跑出 "正确" 结果, 但绝不能依赖
//
// 注意: 为了让 Demo 1 在 x86 上也能**稳定复现错误**, 加上 volatile
//       (volatile 阻止编译器重排, 保留 CPU 重排窗口),
//       并且把 data 用一个结构体包起来强制跨缓存行写入。

struct Payload {
    int v;
    char pad[64] = {0};  // 防止编译器把 v 塞进同一个寄存器 / 缓存行
};

static Payload payload;                          // 改名: 避开 std::data
static volatile bool flag_no_sync = false;        // volatile 不是同步, 只是防编译器优化

static void producer_no_sync() {
    payload.v = 42;
    // 编译器 / CPU 有可能把这一行重排到 payload.v = 42 之前
    flag_no_sync = true;
}

static void consumer_no_sync(int* out) {
    while (!flag_no_sync) {}  // 自旋等
    *out = payload.v;          // **期望 42, 但可能读到 0**
}

static void demo_data_race() {
    cout << "=== Demo 1: 普通变量 (无同步, UB) ===\n";
    int wrong = 0;
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        payload.v = 0;
        flag_no_sync = false;

        int observed = -1;
        thread p(producer_no_sync);
        thread c(consumer_no_sync, &observed);
        p.join();
        c.join();
        if (observed != 42) ++wrong;
    }
    cout << "  " << N << " 次里读到非 42 的次数: " << wrong
         << " (x86 上通常很少, 弱序架构 / 高优化下会非常多)\n";
    cout << "  关键: 这是 UB, 不管看起来「对不对」, 编译器有权编译出任何结果。\n\n";
}

// ============================================================================
// Demo 2: std::atomic, 默认 memory_order_seq_cst
// ============================================================================

static atomic<bool> ready_sc{false};
static Payload data_sc;

static void producer_seq_cst() {
    data_sc.v = 42;
    ready_sc.store(true);  // 默认 seq_cst
}

static void consumer_seq_cst(int* out) {
    while (!ready_sc.load()) {}  // 默认 seq_cst
    *out = data_sc.v;            // 保证读到 42
}

static void demo_seq_cst() {
    cout << "=== Demo 2: std::atomic, 默认 seq_cst ===\n";
    int wrong = 0;
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        data_sc.v = 0;
        ready_sc.store(false);

        int observed = -1;
        thread p(producer_seq_cst);
        thread c(consumer_seq_cst, &observed);
        p.join();
        c.join();
        if (observed != 42) ++wrong;
    }
    cout << "  " << N << " 次里读到非 42 的次数: " << wrong
         << " (期望 0, seq_cst 保证全局顺序一致)\n\n";
}

// ============================================================================
// Demo 3: std::atomic, acquire / release 配对 (经典同步模式)
// ============================================================================

static atomic<bool> ready_ar{false};
static Payload data_ar;

static void producer_acq_rel() {
    data_ar.v = 42;
    // release: 这一行之前的"所有写入"都对之后 acquire 这个值的人可见
    ready_ar.store(true, memory_order_release);
}

static void consumer_acq_rel(int* out) {
    // acquire: 看到 ready_ar==true 时, 同时能看到 release 之前的所有写入
    while (!ready_ar.load(memory_order_acquire)) {}
    *out = data_ar.v;
}

static void demo_acquire_release() {
    cout << "=== Demo 3: std::atomic, acquire / release 配对 ===\n";
    int wrong = 0;
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        data_ar.v = 0;
        ready_ar.store(false, memory_order_relaxed);

        int observed = -1;
        thread p(producer_acq_rel);
        thread c(consumer_acq_rel, &observed);
        p.join();
        c.join();
        if (observed != 42) ++wrong;
    }
    cout << "  " << N << " 次里读到非 42 的次数: " << wrong
         << " (期望 0, release-acquire 形成 synchronizes-with)\n\n";
}

// ============================================================================
// Demo 4: std::atomic, 但 flag 用 relaxed — 翻车现场
// ============================================================================
//
// 这是最常见的认知误区: "我都用 atomic 了, 为啥还能读到旧值?"
// 原因: relaxed 只保证单变量原子性, **不与别的读写同步**。
//       ready_rlx 顺序变成 true 不代表 data_rlx.v = 42 一定先于它被看见。

static atomic<bool> ready_rlx{false};
static Payload data_rlx;

static void producer_relaxed() {
    data_rlx.v = 42;
    ready_rlx.store(true, memory_order_relaxed);  // ⚠️ relaxed, 不发布之前写入
}

static void consumer_relaxed(int* out) {
    while (!ready_rlx.load(memory_order_relaxed)) {}
    *out = data_rlx.v;  // 可能读到 0
}

static void demo_relaxed_break() {
    cout << "=== Demo 4: std::atomic 但用 relaxed, payload 可能旧 ===\n";
    int wrong = 0;
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        data_rlx.v = 0;
        ready_rlx.store(false, memory_order_relaxed);

        int observed = -1;
        thread p(producer_relaxed);
        thread c(consumer_relaxed, &observed);
        p.join();
        c.join();
        if (observed != 42) ++wrong;
    }
    cout << "  " << N << " 次里读到非 42 的次数: " << wrong
         << " (x86 上较少, ARM / RISC-V 上极多; relaxed 不解决可见性)\n\n";
}

int main() {
    demo_data_race();
    demo_seq_cst();
    demo_acquire_release();
    demo_relaxed_break();
    return 0;
}