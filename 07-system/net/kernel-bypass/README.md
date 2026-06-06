# Kernel Bypass Communication

kernel bypass 的目标是让应用尽量绕过传统 Linux socket 协议栈，直接在用户态和网卡队列之间收发 packet。

传统 UDP 收包路径大致是：

```text
NIC -> DMA -> driver -> skb -> IP/UDP stack -> socket buffer -> epoll/recvfrom -> app
```

DPDK 这类 kernel bypass 路径大致是：

```text
NIC -> DMA -> userspace packet buffer -> app polling loop
```

它减少了：

- syscall
- skb 分配和释放
- socket buffer 排队
- qdisc 路径
- 内核协议栈处理
- 线程睡眠和唤醒

代价是应用要自己处理更多事情：

- Ethernet/IP/UDP header
- checksum
- 网卡队列配置
- buffer 生命周期
- CPU 绑核和 NUMA
- 流控、丢包、重传、拥塞控制

## 1. 如何保证低延迟

低延迟不能只靠一个 API，需要端到端控制。

### 减少排队

排队是 tail latency 的主要来源。

需要关注：

```text
NIC RX/TX ring
socket buffer
应用内部队列
风控队列
日志队列
线程调度队列
```

低延迟系统倾向于：

- 小队列
- 固定上限
- 超时丢弃 stale request
- backpressure
- 忙轮询代替睡眠等待

### 减少拷贝

传统路径里 packet 可能被包装成 `skb`，再进入 socket buffer，应用 `recv` 时再复制出来。

kernel bypass 通常使用预分配 packet buffer：

```text
NIC DMA writes packet into mbuf
app reads mbuf directly
app sends mbuf back to NIC TX ring
```

### 减少调度抖动

常见手段：

```text
CPU pinning
NUMA affinity
isolcpus
SCHED_FIFO / SCHED_RR
关闭热路径日志
避免动态内存分配
避免锁竞争
```

如果目标是微秒级延迟，线程睡眠再被唤醒的成本经常太高，所以会使用 busy polling。

### 减少协议栈成本

如果业务只是收发 UDP market data 或交易指令，完整 TCP/IP socket 栈可能不是必须的。

DPDK 可以直接处理 L2/L3/L4 header：

```text
Ethernet header
IPv4 header
UDP header
payload
```

这也是为什么它更快，但也更难写。

## 2. DPDK 通信模型

DPDK 程序通常包含：

```text
EAL init
create mempool
configure port
setup RX queue
setup TX queue
start port
poll RX burst
process packets
send TX burst
```

热路径伪代码：

```text
while running:
    n = rte_eth_rx_burst(port, queue, bufs, BURST)

    for packet in bufs:
        parse ethernet/ip/udp
        build response

    rte_eth_tx_burst(port, queue, bufs, n)
```

这里没有 `recvfrom()`，也没有 `sendto()`。

## 3. 本目录示例

`dpdk_udp_echo.c` 是一个最小 DPDK UDP echo 示例：

- 从单个网卡 port 的 RX queue 轮询收包
- 只处理 IPv4 UDP packet
- 交换源/目的 MAC
- 交换源/目的 IP
- 交换源/目的 UDP port
- 重新计算 IPv4 和 UDP checksum
- 通过 TX queue 发回

它不是完整生产系统，但足够展示 kernel bypass 通信的核心结构。

## 4. 编译

需要系统已安装 DPDK 和 `pkg-config`。

```bash
make
```

如果没有 DPDK，会看到类似：

```text
Package libdpdk was not found
```

## 5. 运行前准备

真实运行需要 root 权限、hugepage 和可被 DPDK 接管的网卡。

常见准备步骤：

```bash
sudo modprobe vfio-pci
sudo dpdk-hugepages.py --setup 1G
sudo dpdk-devbind.py --status
sudo dpdk-devbind.py --bind=vfio-pci <pci-address>
```

运行：

```bash
sudo ./dpdk_udp_echo -l 1 -n 4 -- -p 0
```

其中：

- `-l 1`：使用 CPU core 1
- `-n 4`：内存通道数，按机器情况调整
- `-p 0`：使用 DPDK port 0

## 6. 为什么不能直接在普通环境里验证

DPDK 不是普通用户态 socket 程序，它需要：

- DPDK 开发库
- hugepage
- root 权限
- 支持的 NIC
- 网卡从 Linux network driver 切到 DPDK driver

所以在普通开发容器或沙箱里，通常只能编译或阅读代码，不能真正收发 packet。

## 7. 和 AF_XDP / RDMA 的区别

| 技术 | 特点 |
| --- | --- |
| DPDK | 用户态直接管理 NIC queue，绕过 socket 协议栈，低延迟高吞吐 |
| AF_XDP | 通过 XDP hook 和用户态 ring 收发包，仍依赖 Linux 生态，部署相对温和 |
| RDMA | 远端直接内存访问，常用于 HPC、存储、分布式训练 |

DPDK 更像“应用自己写一个轻量网络栈”。

AF_XDP 更像“在 Linux 体系内把 packet 尽早交给用户态”。

RDMA 更像“绕过 CPU/内核，让 NIC 直接搬运内存”。

## 8. 小结

kernel bypass 的低延迟来自：

```text
少 syscall
少拷贝
少内核协议栈
少睡眠唤醒
少动态分配
用户态直接轮询 NIC queue
```

它适合极致低延迟场景，例如 market data、交易网关、packet gateway、高性能网络服务。

但它也把复杂度从内核转移到了应用。生产系统还必须处理风控、限速、丢包、监控、故障恢复和安全隔离。
