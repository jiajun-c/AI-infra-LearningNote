# 低延时网络学习路线

低延时网络关注的不是平均吞吐，而是一次请求从发出到收到响应的路径里，每一段时间花在哪里，尤其是 P99/P999 tail latency。

可以先抓住三个问题：

```text
数据包在 Linux 里怎么走？
延迟主要来自哪些队列、拷贝、调度和协议行为？
如何观测并逐步减少这些延迟？
```

## 1. 延迟指标

低延时系统不要只看平均值。

常见指标：

```text
avg latency：平均延迟
P50：中位数
P95/P99/P999：尾延迟
jitter：延迟抖动
timeout/retry：超时和重试比例
```

平均值很容易掩盖问题。例如：

```text
99% 请求 100us
1% 请求 10ms
```

平均值可能还不错，但 P99 已经很差。

## 2. Linux 收包路径

简化收包路径：

```text
NIC 收到 packet
  -> DMA 写入内存
  -> 中断 / NAPI polling
  -> 驱动从 RX ring 取包
  -> 构造 skb
  -> 协议栈处理 Ethernet/IP/TCP/UDP
  -> 放入 socket receive buffer
  -> epoll/select/io_uring 通知应用
  -> 应用 recv/read
```

关键词：

```text
NIC
DMA
RX ring
interrupt
NAPI
softirq
skb
socket receive buffer
```

低延时优化时，要特别关注：

```text
中断被哪个 CPU 处理
softirq 是否堆积
应用线程是否和网卡队列在同一个 NUMA node
包是否在 socket buffer / backlog 队列里排队
```

## 3. Linux 发包路径

简化发包路径：

```text
应用 send/write
  -> socket send buffer
  -> TCP/UDP/IP 协议栈封装
  -> qdisc
  -> 网卡 TX ring
  -> DMA
  -> NIC 发包
```

关键词：

```text
send buffer
qdisc
TX ring
TSO/GSO
checksum offload
```

发包低延时常见问题：

```text
send buffer 堆积
qdisc 排队
TCP 小包合并
网卡 interrupt coalescing 参数过大
应用线程频繁 sleep/wakeup
```

## 4. 延迟来源

可以把网络延迟拆成几类：

```text
协议延迟：TCP 握手、重传、拥塞控制、Nagle、Delayed ACK
内核延迟：syscall、skb 分配、协议栈处理、qdisc、softirq
调度延迟：线程睡眠、唤醒、上下文切换、CPU 迁移
队列延迟：socket buffer、backlog、RX/TX ring、应用队列
硬件延迟：NIC、PCIe、DMA、interrupt coalescing、NUMA
应用延迟：锁、内存分配、日志、GC、batch 等待
```

低延时优化的核心是减少排队和不必要的等待。

## 5. TCP 低延时重点

TCP 是可靠传输，但可靠性会带来延迟机制。

需要重点理解：

```text
三次握手：连接建立延迟
慢启动：刚开始发送窗口较小
重传：丢包后延迟可能突然放大
拥塞控制：网络拥塞时主动降速
Nagle：小包合并，可能增加等待
Delayed ACK：接收端延迟 ACK，可能和 Nagle 叠加
TIME_WAIT：短连接大量创建销毁时的资源影响
```

小请求低延时服务中，常见配置：

```cpp
int one = 1;
setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
```

`TCP_NODELAY` 会关闭 Nagle 算法，减少小包等待。但代价是可能发出更多小包，增加网络和 CPU 压力。

## 6. I/O 模型

网络事件处理方式会直接影响延迟。

```text
blocking I/O：
  简单，但一个 fd 可能阻塞整个线程。

nonblocking + epoll：
  常见高并发模型，fd 没数据时返回 EAGAIN，事件循环继续处理其他连接。

io_uring：
  用提交队列和完成队列批量提交 I/O，减少 syscall 和调度开销。

busy polling：
  线程主动轮询，减少睡眠/唤醒延迟，但会增加 CPU 占用。
```

本仓库相关笔记：

```text
../os/batch/README.md
../os/select/README.md
../os/epoil/README.md
../os/io_uring/README.md
```

## 7. batch 和低延时的矛盾

batch 能提高吞吐，因为它摊薄 syscall 和协议栈成本：

```text
sendmmsg
recvmmsg
writev
readv
io_uring submit/completion batch
```

但 batch 也可能增加延迟：

```text
为了凑一批，单个请求要多等一会儿
batch 太大时，后面的请求排队时间变长
```

低延时系统里常见做法：

```text
设置最大 batch size
设置最大等待时间
负载低时小 batch 或不 batch
负载高时适度 batch 保吞吐
```

## 8. 中断、NAPI 和 busy poll

网卡收包有两种直觉模式：

```text
中断：包来了通知 CPU，省 CPU，但唤醒有延迟
轮询：CPU 主动查包，延迟低，但耗 CPU
```

Linux NAPI 把两者结合：

```text
低流量时靠中断
高流量时切到 polling，减少中断风暴
```

低延时可以关注：

```text
interrupt coalescing：网卡攒几个包或等一段时间再中断
busy_poll：应用在 socket 上短时间忙等，减少睡眠唤醒
IRQ affinity：把网卡中断绑到合适 CPU
RSS：多队列网卡把不同 flow 分散到不同 CPU
```

工具：

```bash
ethtool -c eth0      # 查看 interrupt coalescing
ethtool -l eth0      # 查看网卡队列
cat /proc/interrupts
cat /proc/softirqs
```

## 9. 观测工具

低延时优化先观测，不要凭感觉调参数。

连接和 socket：

```bash
ss -tin
ss -u -a
netstat -s
nstat
```

抓包：

```bash
tcpdump -i eth0 -nn host <ip>
tcpdump -i lo -nn port <port>
```

网卡：

```bash
ethtool -S eth0
ethtool -c eth0
ip -s link
```

CPU 和调度：

```bash
pidstat -w -p <pid> 1
perf stat -e context-switches,cpu-migrations,cycles,instructions <cmd>
perf top
```

内核网络路径：

```bash
perf record -g <cmd>
perf report
```

如果使用 eBPF/bcc，可以继续看：

```text
tcptop
tcplife
tcpconnect
tcpaccept
runqlat
softirqs
```

## 10. 优化路线

一个实用顺序：

```text
1. 先测 P50/P99/P999，而不是只看 QPS
2. 用 tcpdump/ss/nstat 判断是否有重传、排队、窗口问题
3. 用 pidstat/perf 看上下文切换和 CPU 热点
4. 减少阻塞 I/O，使用 nonblocking + epoll/io_uring
5. 控制 batch，不让凑批等待伤害 tail latency
6. 绑定 CPU/IRQ，减少 CPU migration 和 NUMA 跨节点
7. 调整网卡 coalescing，权衡 CPU 和延迟
8. 大幅极限优化时再考虑 XDP/DPDK/RDMA
```

## 11. 更高级方向

如果继续往 AI Infra / 分布式训练推理方向走，可以重点学：

```text
RDMA：绕过内核协议栈，远端直接内存访问
RoCE：在以太网上跑 RDMA
InfiniBand：HPC/训练集群常见网络
NCCL 网络路径：GPU 间跨机通信
GPUDirect RDMA：网卡直接访问 GPU memory
XDP/eBPF：在协议栈前处理包
DPDK：用户态轮询网卡，绕过内核协议栈
QUIC：基于 UDP 的现代传输协议
```

## 小结

```text
低延时网络的核心不是单个 API，而是端到端路径：
应用 -> syscall/event loop -> socket buffer -> 协议栈 -> qdisc -> NIC -> 网络 -> 对端。

优化时优先减少：
排队、阻塞、唤醒、上下文切换、跨核迁移、重传、过大的 batch 等待。
```
