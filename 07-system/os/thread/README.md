# Thread

线程是 CPU 调度的基本单位，进程是资源分配的基本单位。

一个进程可以包含多个线程。同一进程内的线程共享进程资源，例如：

- 虚拟地址空间
- 文件描述符表
- 当前工作目录
- 信号处理配置
- 全局变量、堆内存、mmap 区域

每个线程也有自己的私有状态，例如：

- 线程 ID
- 寄存器上下文
- 栈
- errno
- 调度状态

可以简单理解为：

```text
process = resources
thread  = execution flow
```

## 1. 为什么需要线程

如果一个程序只有一个执行流，那么它在等待 I/O、锁、定时器时，整个程序的工作都会停下来。

线程可以让一个进程内部同时存在多个执行流：

```text
main thread:  accept new connection
worker A:     handle request A
worker B:     handle request B
worker C:     wait for disk I/O
```

线程常用于：

- 并发处理多个任务
- 利用多核 CPU 并行执行
- 把阻塞操作和计算操作拆开
- 构建线程池，减少频繁创建和销毁线程的开销

## 2. 线程和进程的区别

| 对比项 | 进程 | 线程 |
| --- | --- | --- |
| 资源 | 拥有独立地址空间 | 共享所属进程的地址空间 |
| 调度 | 可以被内核调度 | 可以被内核调度 |
| 创建开销 | 较大 | 较小 |
| 通信方式 | IPC，如 pipe、socket、共享内存 | 共享变量、锁、条件变量 |
| 崩溃影响 | 通常影响当前进程 | 可能影响整个进程 |

进程之间默认隔离，所以通信成本高但安全性更好。

线程之间共享内存，所以通信成本低，但更容易出现数据竞争。

## 3. 内核线程

内核线程由操作系统直接管理和调度。

Linux 上常用的 `pthread` 在用户视角是线程库 API，但底层通常会通过 `clone()` 创建可以被内核调度的执行实体。

特点：

- 能被内核独立调度
- 可以运行在多个 CPU 核上
- 一个线程阻塞时，不一定阻塞同进程里的其他线程
- 创建、销毁、切换需要进入内核，开销比纯用户态调度更大

适合：

- 多核并行计算
- 阻塞 I/O 场景
- 需要被操作系统公平调度的任务

## 4. 用户线程

用户线程由用户态运行时或线程库管理，内核不一定知道每个用户线程的存在。

特点：

- 创建和切换可以在用户态完成
- 切换开销小
- 调度策略可以由语言运行时自定义
- 如果映射到底层只有一个内核线程，一个用户线程阻塞在系统调用里，可能导致整个线程组无法继续运行

典型模型：

```text
many user threads -> one kernel thread
many user threads -> many kernel threads
```

用户线程的优势是轻量，问题是需要运行时处理阻塞 I/O、抢占、公平性和多核利用。

## 5. 协程

协程比线程更轻量，通常由语言运行时调度。

协程的核心特点是协作式切换：

```text
coroutine A runs
coroutine A await I/O
runtime switches to coroutine B
coroutine B runs
```

协程通常有更小的栈，可以创建大量并发任务。它适合 I/O 密集型场景，例如网络服务、异步任务调度。

注意：

- 协程不是天然并行
- 多个协程要想在多个 CPU 核上同时运行，仍然需要多个内核线程承载
- 协程里的阻塞系统调用如果没有被运行时接管，仍然可能卡住底层线程

## 6. 线程状态

从调度角度看，线程常见状态包括：

```text
new -> runnable -> running -> blocked -> runnable -> running -> exit
```

- `runnable`：可以运行，正在等待 CPU
- `running`：正在 CPU 上执行
- `blocked`：等待 I/O、锁、条件变量、信号量等事件
- `exit`：线程执行结束，等待资源回收

线程切换时，内核需要保存当前线程的寄存器、栈指针、程序计数器等上下文，再恢复另一个线程的上下文。

从阻塞状态获取资源会变成可运行状态

## 7. 线程同步

线程共享地址空间，所以读写共享变量必须考虑并发安全。

常见同步原语：

| 原语 | 用途 |
| --- | --- |
| mutex | 保护临界区，同一时刻只允许一个线程进入 |
| rwlock | 读多写少场景，允许多个读者并发 |
| condition variable | 等待某个条件成立，通常配合 mutex 使用 |
| semaphore | 控制资源数量，支持 wait/post |
| atomic | 对简单变量做原子读改写 |
| barrier | 等待一组线程都到达某个阶段 |

例如 mutex 的基本模式：

```text
lock(mutex)
read/write shared data
unlock(mutex)
```

条件变量的典型模式：

```text
lock(mutex)
while condition is false:
    wait(cond, mutex)
use shared data
unlock(mutex)
```

这里必须用 `while`，因为线程被唤醒后条件不一定仍然成立。

## 8. 数据竞争

数据竞争指多个线程同时访问同一份共享数据，并且至少有一个线程在写，同时缺少同步。

例如：

```text
counter = counter + 1
```

这看起来是一行，但实际可能包含：

```text
load counter
add 1
store counter
```

两个线程并发执行时，可能都读到旧值，最后只加了一次。

解决方式：

- 用 mutex 保护
- 用 atomic 变量
- 避免共享，把数据拆成线程私有
- 用消息队列传递数据所有权

## 9. 线程池

频繁创建线程会有开销，线程池会提前创建一组 worker：

```text
task queue -> worker threads -> execute tasks
```

线程池通常包含：

- 任务队列
- worker 线程
- mutex 保护队列
- condition variable 通知有新任务
- shutdown 标志用于退出

线程池适合大量短任务。它可以减少线程创建开销，也能限制最大并发，避免线程数量失控。

## 10. 常见问题

### 死锁

多个线程互相等待对方释放资源。

```text
thread A holds lock1, waits lock2
thread B holds lock2, waits lock1
```

常见解决方式：

- 固定加锁顺序
- 减小临界区
- 使用 trylock 或超时锁
- 避免在持锁时调用未知外部逻辑

### 竞态条件

程序结果依赖线程执行时序。

例如检查再执行：

```text
if queue is not empty:
    pop queue
```

如果检查和 pop 之间没有锁，另一个线程可能已经把队列取空。

### 伪共享

多个线程修改不同变量，但这些变量落在同一个 cache line 上，导致缓存一致性流量很大。

```text
thread A writes counter_a
thread B writes counter_b
```

如果 `counter_a` 和 `counter_b` 相邻，可能互相拖慢。

### 线程泄漏

创建了线程但没有 `join` 或 `detach`，线程结束后的资源可能无法及时回收。

## 11. 总结

线程让一个进程内部拥有多个执行流。

- 进程负责资源隔离
- 线程负责执行和调度
- 内核线程能利用多核，但切换开销更大
- 用户线程和协程更轻量，但依赖运行时处理阻塞和调度
- 线程共享内存，通信方便，但必须处理数据竞争、死锁和可见性问题

线程的关键不是“能同时跑”，而是“共享资源时如何保持正确”。
