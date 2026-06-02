# epoll demo 和性能测试

`epoll` 是 Linux 上常用的 I/O 多路复用接口，适合大量 fd 的事件驱动服务。

它和 `select` 的核心区别是：

```text
select：
  每次调用都传入整组 fd
  返回后应用还要扫描整组 fd

epoll：
  fd 先注册到内核
  epoll_wait 只返回已经就绪的事件
```

基本流程：

```cpp
int epfd = epoll_create1(0);

epoll_event ev{};
ev.events = EPOLLIN;
ev.data.fd = fd;
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);

epoll_event events[1024];
int n = epoll_wait(epfd, events, 1024, -1);

for (int i = 0; i < n; ++i) {
    int ready_fd = events[i].data.fd;
    read(ready_fd, buffer, sizeof(buffer));
}
```

## 性能测试

本目录的 `epoll_bench.cpp` 用 `pipe()` 做本地测试，方便和 `../select/select_bench.cpp` 对比：

```text
1. 创建 N 个 pipe
2. 把所有 pipe 的读端注册进 epoll
3. 每轮只向最后一个 pipe 写 1 字节
4. epoll_wait 返回就绪事件
5. 只读取 epoll_wait 返回的 fd
```

编译：

```bash
g++ -O2 -std=c++17 epoll_bench.cpp -o epoll_bench
```

运行：

```bash
./epoll_bench 20000 16 64 128 256 384 1024 4096
```

输出字段：

```text
fds       ：注册到 epoll 的 fd 数量
max_fd    ：当前测试中的最大 fd 编号
iterations：循环次数
us/iter   ：每轮平均耗时，包含 write + epoll_wait + read
iter/s    ：每秒能完成多少轮
ready     ：总共处理到的就绪事件数
```

## 预期现象

这个测试每轮只有 1 个 fd 就绪

因此：

```text
select：fd 越多，每轮扫描成本越高
epoll ：只返回 ready fd，fd 总数增加时每轮耗时增长较慢
```

`epoll` 并不是没有成本。注册 fd 时要调用 `epoll_ctl`，内核里也要维护事件结构。但对于“连接很多、活跃连接较少”的服务，它通常比 `select` 更适合。

一次实测结果：

```text
fds             max_fd    iterations       us/iter        iter/s         ready
16                  33         20000          0.92       1082888         20000
64                 129         20000          0.93       1071559         20000
128                257         20000          0.92       1089646         20000
256                513         20000          0.94       1065449         20000
384                769         20000          0.92       1084540         20000
1024              2049         20000          0.93       1075718         20000
4096              8193         20000          0.92       1088545         20000
```

和 `select` 的对比很明显：同样每轮只有 1 个 fd 就绪时，`epoll_wait` 只返回这 1 个 ready fd，不需要应用层扫描全部 fd。

## LT 和 ET

`epoll` 有两种常见触发模式：

```text
LT，Level Trigger：
  只要 fd 仍然可读，下次 epoll_wait 还会继续通知。
  更容易写对。

ET，Edge Trigger：
  状态从不可读变为可读时通知一次。
  通常要配合 nonblocking fd，并一次读到 EAGAIN。
  性能可能更好，但更容易漏事件。
```

LT和ET的区别在于比如有个被监听的fd的时候，假设本次有100个字节的输出，LT每次拿10个，后面会一直拿直到90个没有了，但是对于ET模式的，如果第二次没有新的数据产生了，那么就不会进行读取

这个 demo 使用默认 LT 模式。

## nonblocking + ET demo

`nonblocking_et_demo.cpp` 演示两个点：

```text
1. nonblocking fd 在没有数据时不会睡眠，而是立刻返回 EAGAIN
2. EPOLLET 模式下，收到事件后应该循环 read 到 EAGAIN
```

编译：

```bash
g++ -O2 -std=c++17 nonblocking_et_demo.cpp -o nonblocking_et_demo
```

运行：

```bash
./nonblocking_et_demo
```

预期现象：

```text
空 pipe + nonblocking read：
  read 立刻返回 -1 / EAGAIN

EPOLLET 只读 1 字节：
  第一次 epoll_wait 有事件
  只读走 1 字节后，pipe 里仍有数据
  第二次 epoll_wait 可能超时，因为没有新的边缘变化

EPOLLET 读到 EAGAIN：
  一次事件里把已有数据读干净
  read 返回 EAGAIN，表示可以停下来等下一次事件
```

这就是 ET 模式常见写法的原因：

```cpp
while (true) {
    ssize_t n = read(fd, buffer, sizeof(buffer));
    if (n > 0) {
        // process data
        continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        break;
    }
}
```
