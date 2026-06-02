# select demo 和性能测试

`select` 是早期 Unix/Linux I/O 多路复用接口。它可以让一个线程同时等待多个 fd：

```text
应用把一组 fd 放进 fd_set
调用 select()
内核检查哪些 fd 可读/可写/异常
select 返回后，应用再扫描 fd_set 找到就绪 fd
```

核心接口：

```cpp
int select(
    int nfds,
    fd_set* readfds,
    fd_set* writefds,
    fd_set* exceptfds,
    struct timeval* timeout);
```

其中 `nfds` 是最大 fd 值加 1。也就是说，`select` 关注的不只是 fd 数量，还和最大 fd 编号有关。

## 基本使用模式

```cpp
fd_set read_set;
FD_ZERO(&read_set);
FD_SET(fd, &read_set);

int ready = select(fd + 1, &read_set, nullptr, nullptr, nullptr);

if (ready > 0 && FD_ISSET(fd, &read_set)) {
    read(fd, buffer, sizeof(buffer));
}
```

注意：`select` 会修改传入的 `fd_set`，所以循环里通常要保留一份 `base_set`，每次调用前复制：

```cpp
fd_set read_set = base_set;
select(max_fd + 1, &read_set, nullptr, nullptr, nullptr);
```

## select 的主要问题

```text
1. fd_set 大小有限，常见 FD_SETSIZE=1024
2. 每次调用都要从用户态拷贝 fd_set 到内核
3. select 返回后，应用还要从头扫描 fd_set
4. nfds 依赖最大 fd 编号，fd 编号越大扫描范围越大
```

所以连接数多时，`select` 的扩展性不好。后来的 `poll` 去掉了固定 `FD_SETSIZE` 限制，但仍然需要线性扫描；`epoll` 则把 fd 注册到内核，只返回就绪事件。

## 性能测试

本目录的 `select_bench.cpp` 用 `pipe()` 做本地测试，不依赖网络：

```text
1. 创建 N 个 pipe
2. select 监听所有 pipe 的读端
3. 每轮只向最后一个 pipe 写 1 字节
4. select 返回后扫描所有 fd，找到可读 fd 并读走
5. 统计每轮 select + scan + read 的平均耗时
```

编译：

```bash
g++ -O2 -std=c++17 select_bench.cpp -o select_bench
```

运行默认测试：

```bash
./select_bench
```

指定迭代次数和 fd 数量：

```bash
./select_bench 20000 16 64 128 256 384
```

输出字段：

```text
fds       ：监听的 fd 数量
max_fd    ：select 的 nfds 接近 max_fd + 1
iterations：循环次数
us/iter   ：每轮平均耗时，包含 write + select + scan + read
iter/s    ：每秒能完成多少轮
ready     ：总共处理到的就绪事件数
```

预期现象：

```text
fds 越多，us/iter 越高
这是因为 select 返回后，用户态仍然要扫描 fd_set
```

这个 demo 每轮只有 1 个 fd 就绪，但仍然要扫描所有被监听的 fd。真实服务中如果大量连接多数时间不活跃，这个成本会很明显。

如果 fd 数量太大，可能看到：

```text
fd xxxx exceeds FD_SETSIZE=1024
```

这是因为 `select` 的 `fd_set` 有固定大小限制。这个 demo 使用 pipe，一组 pipe 会占用读端和写端两个 fd，所以能测试的 pipe 数量会小于 `FD_SETSIZE`。

## 和 epoll 的区别

```text
select：
  每次传入整组 fd
  每次返回后应用扫描整组 fd
  适合 fd 少、简单可移植的场景

epoll：
  fd 先注册到内核
  epoll_wait 只返回就绪事件
  适合大量连接、高并发服务
```
