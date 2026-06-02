# batch 请求

batch 的核心思想是：不要每个请求都单独进一次内核，而是一次系统调用处理多个请求。

在网络发送里，最直接的对比是：

```text
send：每个 UDP message 调用一次系统调用
sendmmsg：一次系统调用发送多个 UDP message
```

系统调用本身有固定开销：

```text
用户态/内核态切换
参数检查
socket 查找和锁
协议栈处理
调度和唤醒
```

如果每个请求都很小，系统调用开销会占很大比例。batch 可以把这部分固定成本摊薄。

## 测试代码

本目录的 `udp_batch_bench.cpp` 用 UDP localhost 测试吞吐：

```text
一个接收线程负责 recv 并丢弃数据
发送端先用 send 逐条发送
发送端再用 sendmmsg 批量发送
最后比较 messages/sec 和 MiB/sec
```

编译：

```bash
g++ -O2 -std=c++17 udp_batch_bench.cpp -o udp_batch_bench -pthread
```

运行：

```bash
./udp_batch_bench
```

参数：

```bash
./udp_batch_bench <messages> <payload_bytes> <batch_size>
```

例如：

```bash
./udp_batch_bench 1000000 64 32
```

输出类似：

```text
UDP localhost throughput, payload=64 bytes, messages=1000000, batch=32

mode            messages       seconds        Mmsg/s         MiB/s
send             1000000          1.23          0.81         49.60
sendmmsg         1000000          0.72          1.39         84.77
```

实际数字会随 CPU、内核版本、虚拟化环境、socket buffer、后台负载变化。

## 为什么 batch 会更快

逐条 `send`：

```text
message 1 -> syscall
message 2 -> syscall
message 3 -> syscall
...
```

`sendmmsg`：

```text
message 1 \
message 2  \
message 3   -> one syscall
...        /
message N /
```

当 payload 很小时，吞吐瓶颈经常不是拷贝多少字节，而是每秒能承受多少次系统调用和协议栈路径开销。

## 代价

batch 不是越大越好。

```text
batch 太小：摊薄系统调用开销不明显
batch 太大：等待凑批会增加延迟，可能恶化 tail latency
```

所以实际服务里通常要在吞吐和延迟之间折中。
