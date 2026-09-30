# NUMA

## 1. NUMA硬件拓扑

NUMA（Non-Uniform Memory Access，非一致内存访问）是一种多路服务器内存架构。每颗 CPU Socket 拥有本地内存控制器和本地内存，同时也能通过 UPI、xGMI 等片间互联访问其他 Socket 的内存。

```text
┌──────────────── NUMA Node 0 ────────────────┐
│ CPU 0-63        Memory 0        GPU 0-3     │
└──────────────────────┬──────────────────────┘
                       │ UPI / xGMI
┌──────────────────────┴──────────────────────┐
│ CPU 64-127      Memory 1        GPU 4-5     │
└──────────────── NUMA Node 1 ────────────────┘
```

Node 0 的 CPU 能够访问 Memory 1，但请求需要穿过片间互联：

```text
本地访问：CPU Node 0 -> Memory Node 0
远端访问：CPU Node 0 -> Socket Interconnect -> Memory Node 1
```

远端访问通常延迟更高、带宽更低，还会占用 Socket 间互联。具体差距与 CPU 架构、内存通道、BIOS 的 NPS/SNC 配置和访问模式有关，需要在目标机器上实测。

NUMA Node 不一定等于物理 Socket。某些 CPU 可以通过 BIOS 配置把一颗 Socket 划分成多个 NUMA Node，所以应读取操作系统暴露的实际拓扑，而不能只看 CPU 数量。

常用拓扑命令：

```bash
lscpu -e=CPU,NODE,SOCKET,CORE
numactl --hardware
cat /sys/devices/system/node/node*/cpulist
nvidia-smi topo -m
```

分析 NUMA 问题时，需要同时画出三张图：

1. CPU 属于哪个 NUMA Node；
2. 物理内存页位于哪个 NUMA Node；
3. GPU、网卡、NVMe 连接到哪个 PCIe Root Complex 和 NUMA Node。

只检查 CPU 绑核，或者只检查内存位置，都不足以描述完整的数据路径。

## 2. NUMA中的四种局部性

### 2.1 CPU局部性

CPU 局部性描述线程当前在哪个 CPU 上运行，以及它允许在哪些 CPU 上运行。

```bash
taskset -c 0-15 ./app
numactl --cpunodebind=0 ./app
```

C/C++ 可以使用 `sched_setaffinity()` 或 `pthread_setaffinity_np()`。CPU affinity 只限制运行位置，不会自动把线程访问的物理页搬到同一个 Node。

需要区分三个概念：

```text
allowed CPUs：线程被允许去哪里
current CPU：线程此刻在哪里运行
preferred NUMA node：内核认为线程更适合在哪里运行
```

### 2.2 内存局部性

内存局部性描述虚拟地址背后的物理页位于哪个 Node。同一个进程中可以同时存在 Node 0 的匿名页、Node 1 的文件页以及分散在多个 Node 的共享内存。

因此，“进程位于哪个 Node”并不严谨。线程有运行位置，每一组物理页也有各自的位置。

### 2.3 设备局部性

GPU、网卡和 NVMe 通常连接在某个 Socket 的 PCIe Root Complex 下。从 Node 1 的内存向挂在 Node 0 的 GPU 传输数据，可能经过：

```text
Memory 1 -> Node 1 memory controller
         -> Socket interconnect
         -> Node 0 PCIe root complex
         -> GPU
```

GPU 推理服务中的本地化通常希望满足：

```text
加载线程所在 Node
        = pinned memory 所在 Node
        = GPU 所在 Node
```

### 2.4 调度局部性

未绑核线程可以使用多个 Node 的 CPU，但调度器不一定把它们均匀分散。Linux 自动 NUMA 均衡会根据内存访问历史，为线程形成 NUMA 放置偏好。

这意味着“线程可以去 Node 1”和“线程实际上会去 Node 1”是两回事。

## 3. 物理页的初始位置

### 3.1 虚拟内存申请不等于物理页分配

`malloc()` 或匿名 `mmap()` 通常先获得一段虚拟地址，物理页往往在首次访问导致缺页时才真正分配。

```cpp
char* buffer = static_cast<char*>(malloc(size));
// 此时不一定已经为整段 buffer 分配物理页。

memset(buffer, 0, size);
// 写入每个页面，引发缺页并建立物理页。
```

### 3.2 First-touch

默认策略下，匿名页通常遵循 first-touch：第一次真正访问页面的线程运行在哪个 Node，页面就优先从哪个 Node 分配。

```text
线程在 Node 0 上执行 memset
              ↓
页面首次缺页
              ↓
物理页优先分配到 Node 0
```

由此可得：

- 调用 `malloc()` 的线程不一定决定物理页位置；
- 执行初始化循环的线程通常更重要；
- 并行初始化可以让页面分散到多个 Node；
- 初始化后再迁移线程，会留下远端页面；
- first-touch 只解释初始位置，不保证页面以后不迁移。

### 3.3 Page cache

读取模型文件时，典型路径是：

```text
storage -> page cache -> user buffer -> pinned buffer -> GPU
```

同一个文件的读者通常共享 page cache 页。文件页最初由哪个 Node 上的缺页、读取或预读路径建立，会影响它的 NUMA 分布。内核不会仅仅因为两个 Node 都在读取文件，就永久维护两份完全相同的 page cache。

### 3.4 Pinned Memory

Pinned Memory 只是被锁定、不能换出并适合 DMA，不代表它脱离了 NUMA。它背后仍是位于某个 Node 的物理页。

```text
是否 pinned             -> 页面能否稳定用于 DMA
位于哪个 Node            -> CPU 和设备访问路径是否本地
由哪个线程 first-touch   -> 初始物理页位置
```

## 4. Linux内存策略

Linux 可以通过 `set_mempolicy()`、`mbind()` 或 `numactl` 设置内存策略。

| 策略 | 语义 | 常见用途 |
|---|---|---|
| `MPOL_DEFAULT` | 使用默认策略 | 交给系统默认规则处理 |
| `MPOL_LOCAL` | 优先在触页线程当前 Node 分配 | 保留 first-touch 风格并显式指定策略 |
| `MPOL_PREFERRED` | 优先指定 Node，失败时允许回退 | 有首选位置，但不要求绝对绑定 |
| `MPOL_BIND` | 只从指定 Node 集合分配 | 严格控制物理页位置 |
| `MPOL_INTERLEAVE` | 在多个 Node 间轮流分配页面 | 分摊大型顺序访问的内存带宽 |

常见用法：

```bash
numactl --cpunodebind=0 --membind=0 ./app
numactl --preferred=0 ./app
numactl --interleave=all ./app
```

注意：

- CPU affinity 与 memory policy 是两套独立约束；
- `MPOL_BIND` 可能在目标 Node 内存不足时导致分配失败；
- `MPOL_PREFERRED` 允许回退，不保证所有页都位于首选 Node；
- `set_mempolicy()` 主要影响调用线程后续的内存分配；
- `mbind()` 作用于指定虚拟地址范围；
- 改变策略不一定会迁移已经存在的页面；
- `numactl` 的版本差异可能影响选项行为，应结合 `numactl --show` 和 `/proc/<pid>/numa_maps` 验证实际效果。

## 5. Linux自动NUMA均衡

程序不一定了解 NUMA，线程也可能被普通调度器迁移，于是会出现：

```text
线程在 Node 1 运行
        ↓
频繁访问 Node 0 的页面
        ↓
持续产生远端内存访问
```

Linux 自动 NUMA 均衡（Automatic NUMA Balancing）尝试根据运行时访问模式修复这种错位，主要有两个方向：

1. 把页面迁移到访问它的线程附近；
2. 把线程迁移到它经常访问的页面附近。

检查是否开启：

```bash
cat /proc/sys/kernel/numa_balancing
```

它是反馈控制机制：先采样访问，再根据历史决定页面和线程的放置，而不是预先知道应用的最佳布局。

## 6. 自动NUMA均衡如何采样访问

不同内核版本的细节会变化，但核心过程可以抽象为：

```text
周期性扫描任务的一部分虚拟地址空间
                    ↓
修改部分页表项，使后续访问可被内核观察
                    ↓
线程访问页面，产生 NUMA hint fault
                    ↓
记录线程运行 Node 与页面所在 Node
                    ↓
更新线程、页面和 NUMA group 的访问统计
                    ↓
决定迁移页面、迁移线程或暂时不动
```

NUMA hint fault 是内核主动制造的采样事件，通常不表示页面真的丢失，也不是程序访问非法地址。它帮助内核知道：

```text
哪个线程
在什么 Node 上运行
访问了位于哪个 Node 的页面
```

可以查看系统级累计计数：

```bash
grep -E 'numa_(pte_updates|hint_faults|hint_faults_local|pages_migrated)' \
    /proc/vmstat
```

| 指标 | 含义 |
|---|---|
| `numa_pte_updates` | 为 NUMA 采样而处理或更新的页表项 |
| `numa_hint_faults` | 采样触发的 NUMA hint fault |
| `numa_hint_faults_local` | hint fault 发生时，访问已经是本地的 |
| `numa_pages_migrated` | 自动 NUMA 机制迁移的页面数 |

这些值通常是系统启动以来的累计量，单个绝对值意义有限。分析一次实验时，应在实验前后采样并计算增量。

计数很大只表示机制活跃，不能单独证明它是性能瓶颈。还要验证页面迁移流量、线程排队时间和业务延迟之间是否存在因果关系。

## 7. 页面迁移还是线程迁移

假设线程运行在 Node 1，却频繁访问 Node 0 的页面：

```text
CPU Node 1 ---- remote access ----> Memory Node 0
```

### 7.1 迁移页面

```text
Before: CPU Node 1 -> Memory Node 0
After:  CPU Node 1 -> Memory Node 1
```

页面迁移更适合以下情况：

- 页面主要由 Node 1 上的线程访问；
- 页面不是大量线程共享的；
- Node 1 有足够内存；
- 访问模式稳定；
- 搬运页面的成本小于未来远端访问的成本。

### 7.2 迁移线程

```text
Before: CPU Node 1 -> Memory Node 0
After:  CPU Node 0 -> Memory Node 0
```

线程迁移更适合以下情况：

- 热数据大量集中在 Node 0；
- 一个线程访问的数据规模很大；
- 页面被多个线程共享，不适合来回迁移；
- 移动一个线程比移动大量页面便宜；
- CPU affinity 允许线程进入 Node 0。

因此，自动 NUMA 均衡并不只是“发现远端访问就迁移页面”，它也会通过调度器改变线程运行位置。

## 8. NUMA Group与共享数据

多线程程序中的线程经常访问同一批共享页面。如果每个线程只按自己的瞬时采样独立迁移，线程和页面可能来回震荡。

内核可以根据共享访问关系，把相关任务组织成 NUMA group，并聚合访问统计。直觉上，它在判断：

```text
这批线程是不是共同使用同一个工作集？
如果是，它们作为整体更适合放在哪个 Node？
```

这能提高共享内存程序的稳定性，却也可能放大聚集效应：如果大量线程共同读取 Node 0 上的模型参数或 page cache，整个线程组都可能表现出对 Node 0 的偏好。

某些内核版本可以在线程调度信息中观察首选节点：

```bash
grep numa_preferred_nid /proc/<pid>/task/<tid>/sched
```

字段是否存在以及格式如何，取决于内核版本和配置。

## 9. 为什么不同Node的CPU利用率不一致

这里容易混淆两套目标不同的机制。

### 9.1 普通CPU负载均衡

普通调度负载均衡关注 CPU 运行队列和计算负载，倾向于把任务分散到空闲 CPU：

```text
Node 0：80 个 runnable tasks
Node 1：10 个 runnable tasks

倾向：把部分任务从 Node 0 移到 Node 1
```

### 9.2 NUMA局部性优化

自动 NUMA 均衡关注内存局部性：

```text
线程主要访问 Node 0 的页面

倾向：让线程靠近 Node 0
```

两者可能提出相反的方向：

```text
CPU 负载均衡：Node 0 太忙，线程应该去 Node 1
NUMA 局部性：数据在 Node 0，线程应该留在 Node 0
```

调度器需要综合考虑：

- 两个 Node 的 CPU 负载；
- 线程的 NUMA 访问统计和 preferred node；
- 迁移后增加的远端访问成本；
- cache 热度与线程迁移成本；
- CPU affinity、cpuset 和调度域限制；
- 目标 CPU 是否适合接收任务。

调度器不是离线的全局最优化器，而是根据局部统计持续进行启发式决策。系统可能稳定在这种状态：

```text
Node 0：CPU 很忙，但大部分内存访问是本地的
Node 1：CPU 较空闲，但把线程放过去会增加远端访问
```

对于内存受限的吞吐任务，这可能是合理结果；对于依赖 Node 0 CPU 的延迟敏感线程，这可能非常糟糕。

## 10. 线程向热数据扎堆的因果链

假设有 90 个未绑核计算线程，初始时由普通调度器大致平均分布：

```text
Node 0：45 个线程
Node 1：45 个线程
```

如果它们都频繁读取 Node 0 上的共享热数据：

```text
热数据通过 first-touch 或 page cache 集中到 Node 0
                         ↓
Node 1 上的线程远端访问 Node 0
                         ↓
NUMA hint fault 记录访问关系
                         ↓
共享页面迁移收益低，或者移动成本高
                         ↓
内核逐渐为线程建立 Node 0 偏好
                         ↓
更多可迁移线程在 Node 0 上运行
                         ↓
Node 0 CPU 更忙，Node 1 CPU 更空闲
```

最终可能变成：

```text
Node 0：78 个线程，CPU busy 接近 100%
Node 1：12 个线程，CPU 仍有大量空闲
```

如果后续初始化也由 Node 0 上的线程完成，还可能形成正反馈：

```text
Node 0 上的数据更多
        ↓
更多线程偏向 Node 0
        ↓
更多 first-touch 发生在 Node 0
        ↓
新页面继续落在 Node 0
```

CPU 利用率不一致不是自动 NUMA 均衡的直接目标，而是“优化内存局部性”在共享热数据布局下产生的结果。

## 11. 为什么绑核任务反而变慢

假设模型加载线程被绑定到 Node 0，原本是为了建立本地数据路径：

```text
Node 0 Loader
    -> Node 0 page/pinned memory
    -> Node 0 PCIe root complex
    -> Node 0 GPU
```

但普通计算线程没有绑核。自动 NUMA 均衡把大量普通线程吸引到 Node 0 后，双方处境不同：

```text
普通计算线程：可以去 Node 1，但因内存局部性偏向 Node 0
Loader：CPU affinity 只允许 Node 0，无法去 Node 1
```

Node 0 的 runqueue 变长后，Loader 即使只是执行 `memcpy()`，也必须等待 CPU 时间片。它的延迟可以拆成：

```text
墙钟时间 = CPU 实际运行时间 + runqueue 等待时间 + 其他阻塞时间
```

如果 `memcpy()` 实际 CPU 运行时间变化不大，而 runqueue 等待时间显著增加，那么慢因不是复制指令本身，而是线程抢不到 CPU。

> 绑核保证了 Loader 不会跑到远端 Node，但也让它无法逃离已经拥塞的本地 Node。

## 12. 为什么显式MPOL_LOCAL可能改善问题

`MPOL_DEFAULT` 和 `MPOL_LOCAL` 的初始页面放置都可能表现为“在触页线程当前 Node 分配”，但二者并不等价：

- `MPOL_DEFAULT` 表示采用默认内存策略；
- `MPOL_LOCAL` 是显式设置的本地分配策略；
- 内存区域是否继续参与自动 NUMA 均衡，取决于策略、模式标志和内核版本。文章对应的 Linux 5.4 环境中，显式 `MPOL_LOCAL` 抑制了相关自动 NUMA 行为；Linux 5.12 起还支持用 `MPOL_F_NUMA_BALANCING` 配合 `MPOL_BIND` 显式允许自动均衡。

所以设置 `MPOL_LOCAL` 后，性能改善不一定来自“页面终于放到了正确 Node”，也可能来自：

```text
显式内存策略
      ↓
相关地址范围不再以相同方式参与自动 NUMA 均衡
      ↓
线程不再积累相同的 NUMA 放置偏好
      ↓
普通 CPU 负载均衡重新占据主导
      ↓
未绑核线程重新分散到两个 Node
```

一个很有价值的对照实验是：

```text
MPOL_PREFERRED(Node 0) 有效
MPOL_PREFERRED(Node 1) 也有效
```

两个方向相反的 preferred 策略都能改善性能，说明关键变量很可能不是“页面最终位于哪个 Node”，而是“设置显式策略改变了自动 NUMA 行为”。

这体现了性能分析的重要方法：不要只按 API 名称猜测原因，要用方向相反的实验拆分变量。

## 13. 在线推理服务案例

模型快速上下线通常包含以下数据流：

```text
模型文件
   ↓
page cache
   ↓ CPU memcpy
pinned host memory
   ↓ PCIe DMA
GPU memory
```

考虑一台双路服务器：

- Node 0 挂载 4 张 GPU；
- Node 1 挂载 2 张 GPU；
- 模型文件的 page cache 或主进程热数据偏向 Node 0；
- 模型加载线程按 GPU 所在 Node 绑核；
- 大量计算线程没有绑核；
- 自动 NUMA 均衡开启。

即使 Loader、pinned memory 和 GPU 已经按 Node 对齐，Node 0 的模型加载仍可能更慢：

```text
Node 0 热数据
    -> 吸引未绑核计算线程
    -> Node 0 CPU 接近满载
    -> 绑定在 Node 0 的 Loader 排队
    -> Node 0 模型加载延迟上升
```

这说明 NUMA-aware 不只是把关键线程、内存和 GPU 放到一起，还要避免其他任务耗尽同一个 Node 的 CPU 资源。

## 14. 系统化诊断流程

### 14.1 画出硬件拓扑

```bash
lscpu -e=CPU,NODE,SOCKET,CORE
numactl --hardware
cat /sys/devices/system/node/node*/cpulist
nvidia-smi topo -m
```

需要明确：

- 每个 Node 包含哪些 CPU；
- 每个 Node 有多少内存；
- GPU 和网卡挂在哪个 Node；
- Node 间通过什么链路连接；
- 是否存在一个 Socket 被划分成多个 Node 的情况。

### 14.2 对业务链路分段计时

不要只记录“模型加载总时间”，应拆分：

```text
文件读取或 page cache 命中
page cache -> 普通用户内存
普通内存 -> pinned memory
pinned memory -> GPU
GPU 侧模型初始化
```

确定变慢阶段后，再判断它是 CPU 执行变慢、内存路径变慢，还是调度等待增加。

### 14.3 按Node观察CPU

```bash
mpstat -P ALL 1
numastat
```

例如：

```text
Node 0 CPU busy：99%
Node 1 CPU busy：27%
整机平均 busy：63%
```

整机平均值看起来仍有余量，但绑定在 Node 0 的任务已经处于严重竞争状态。NUMA 机器上，整机平均 CPU 利用率经常掩盖局部拥塞。

### 14.4 检查线程运行位置

```bash
ps -L -p <pid> -o pid,tid,psr,pcpu,stat,comm
taskset -pc <tid>
```

`psr` 表示线程最近运行的 CPU。把 CPU 映射回 NUMA Node 后，可以统计线程分布。

单次 `psr` 只是瞬时结果。严谨分析应周期采样，或者使用调度跟踪工具统计一段时间内线程在各 Node 上的 CPU 时间。

### 14.5 判断关键线程是在执行还是排队

```bash
cat /proc/<pid>/task/<tid>/schedstat
```

常见内核中，前三个字段通常是：

```text
累计实际运行时间（ns）
累计 runqueue 等待时间（ns）
累计获得 CPU 的次数
```

应在实验前后读取并计算增量：

```text
runtime_delta = runtime_after - runtime_before
wait_delta    = wait_after - wait_before

排队比率 = wait_delta / runtime_delta
```

解释方式：

- `runtime_delta` 增加：任务可能做了更多工作，或执行本身变慢；
- `wait_delta` 大幅增加：线程主要在等待 CPU；
- wall time 增加但 runtime 接近不变：优先调查调度竞争。

字段含义可能随内核配置变化，使用前应结合当前系统文档确认。

### 14.6 检查物理页位置

```bash
numastat -p <pid>
head -n 20 /proc/<pid>/numa_maps
```

`numa_maps` 可以看到各虚拟内存区域在不同 Node 上的页面数量：

```text
... anon=1024 dirty=1024 active=0 N0=900 N1=124 kernelpagesize_kB=4
```

分析时要区分匿名内存、文件映射、heap、stack、共享内存和性能关键 buffer。进程总内存分布正常，不代表关键热页的位置正常。

### 14.7 检查自动NUMA均衡活动

```bash
cat /proc/sys/kernel/numa_balancing

grep -E 'numa_(pte_updates|hint_faults|hint_faults_local|pages_migrated)' \
    /proc/vmstat

grep -E 'numa_preferred_nid|nr_migrations' \
    /proc/<pid>/task/<tid>/sched
```

重点不是某个数字看起来很大，而是比较不同场景下的增量：

```text
默认策略 vs MPOL_LOCAL
问题发生前 vs 问题发生后
有后台计算线程 vs 没有后台计算线程
Node 0 Loader vs Node 1 Loader
```

## 15. 用数量级排除错误方向

### 15.1 页面迁移是否耗尽带宽

迁移流量可以粗略估算为：

```text
numa_pages_migrated 增量 × 页面大小
```

假设迁移了 90 万个 4 KiB 页面：

```text
900000 × 4096 B ≈ 3.43 GiB
```

如果实验持续十几分钟，业务自身搬运了数百 GiB，那么几 GiB 页面迁移通常不足以解释数倍延迟，应继续检查调度等待。

### 15.2 NUMA扫描是否直接消耗大量CPU

可以结合以下信息判断：

- `numa_pte_updates` 增量；
- perf 中的内核热点；
- 线程内核栈；
- TLB shootdown 相关事件；
- 进程 CPU time 与 wall time 的差异。

扫描、页表修改和 TLB 失效确实有成本，但“机制存在”不等于“它是主瓶颈”。必须验证数量级能否解释业务退化。

### 15.3 远端内存慢还是线程没运行

远端内存是主因时，通常会看到：

- 线程拥有充足的 CPU runtime；
- 本地与远端 memcpy 带宽存在明显差异；
- 页面位置改变后，执行时间随之改变。

调度竞争是主因时，通常会看到：

- 关键线程 runqueue wait 明显增加；
- Node 间 CPU 利用率严重不均；
- 关键线程 runtime 变化不大；
- 移除后台线程后，性能差异消失。

## 16. 控制变量实验

NUMA 问题中多个机制会同时变化，因此每轮只改变一个维度。

| 实验 | 保持不变 | 改变变量 | 回答的问题 |
|---|---|---|---|
| A | 工作负载、内存策略 | Loader CPU Node | 是否只有某个 Node 慢 |
| B | CPU affinity、工作负载 | 内存策略 | 显式策略是否改变行为 |
| C | 页面位置、Loader | 后台线程数量 | 是否由 CPU 竞争触发 |
| D | 工作负载、策略 | 自动 NUMA 开关 | 自动 NUMA 是否进入因果链 |
| E | Loader、线程数 | 数据 first-touch Node | 聚集是否跟随热数据反转 |
| F | CPU 和内存位置 | GPU 所在 Node | PCIe/设备拓扑是否是瓶颈 |

建议比较：

```text
MPOL_DEFAULT
MPOL_LOCAL
MPOL_PREFERRED(Node 0)
MPOL_PREFERRED(Node 1)
MPOL_BIND(Node 0/1)
```

每轮同时记录：

- 业务延迟；
- 每个 Node 的 CPU busy；
- 关键线程 runtime 和 runqueue wait；
- 线程在各 Node 上的 CPU 时间；
- NUMA hint fault 和页面迁移增量；
- 关键 VMA 的页面分布。

## 17. 最小复现方法

一个能复现“线程向热数据扎堆”的实验需要三个角色：

1. 一大块主要位于 Node 0 的共享数据；
2. 大量未绑核、反复读取这块数据的后台线程；
3. 一个绑定在 Node 0、对调度延迟敏感的观测线程。

实验步骤：

```text
1. 在 Node 0 first-touch 一块大内存。
2. 创建足以产生 CPU 压力的未绑核 worker。
3. 让 worker 持续读取共享内存。
4. 在 Node 0 和 Node 1 各运行相同的观测任务。
5. 采样每个 Node 的 CPU busy 和 worker 分布。
6. 比较默认策略与显式内存策略。
7. 移除 worker，确认不对称是否消失。
```

一个好的最小复现应该继续回答：

- 没有共享热数据时是否还会聚集？
- 没有后台线程时 Loader 是否仍然不对称？
- 把热数据改到 Node 1 后，聚集方向是否反转？
- 只设置显式策略、不改变初始页位置时，现象是否缓解？

现象能随热数据位置反转，是比一次性能数字更强的因果证据。

## 18. 解决方案及适用边界

### 18.1 使用显式内存策略

可以在进程启动时使用 `MPOL_LOCAL`、`MPOL_PREFERRED` 或 `MPOL_BIND`。优点是作用范围可控；风险是可能改变页面位置、回退行为以及内存不足时的表现。

不要只验证平均性能，还要覆盖：

- 目标 Node 内存接近耗尽；
- 模型反复上下线；
- 多个模型并发加载；
- 服务长期运行后的稳态；
- 容器权限和不同内核版本。

### 18.2 隔离延迟敏感线程

通过 CPU affinity、cpuset 或容器 CPU 集合，为 Loader、通信线程等预留 CPU：

```text
Node 0 普通 worker：CPU 0-47
Node 0 Loader：CPU 48-55
Node 0 系统/中断：CPU 56-63
```

只给 Loader 绑核，却允许所有普通线程使用 Loader 的 CPU 集合，并不是真正的 CPU 隔离。隔离时还要考虑中断、内核线程和容器 cpuset。

### 18.3 按Node拆分进程

为每个 Node 或每组 GPU 启动独立进程，使其拥有独立的：

- CPU affinity；
- 内存策略；
- pinned memory pool；
- 模型加载线程；
- worker 数量和生命周期。

进程级拆分通常比在一个巨大共享地址空间中协调所有线程更容易推理，但会增加进程间通信与资源管理成本。

### 18.4 复制只读热数据

如果热数据只读且内存容量允许，可以由应用在每个 Node 维护一份副本，让各 Node 的线程访问本地副本。操作系统不会自动为任意共享页提供这种应用语义。

代价包括：

- 增加内存占用；
- 数据更新时要维护一致性；
- page cache 或文件映射场景不一定容易直接复制。

### 18.5 全局关闭自动NUMA均衡

```bash
sysctl kernel.numa_balancing=0
```

这适合作为诊断实验，但会影响整台机器的工作负载。除非已验证所有重要服务，否则不应仅凭一个进程的收益就永久关闭。

### 18.6 自动NUMA均衡并非总是有害

以下场景中它可能有明显收益：

- 应用完全不了解 NUMA；
- 线程长期访问稳定、可迁移的私有工作集；
- 远端内存访问是主要瓶颈；
- 页面或线程迁移后能够长期保持局部性；
- 没有严格的延迟敏感 CPU 隔离需求。

正确目标不是一律关闭自动机制，而是根据工作负载选择：

```text
让页面跟着线程走
让线程跟着页面走
由应用显式管理两者
复制数据以同时获得局部性
```

## 19. 常见误区

### 19.1 绑核后内存一定是本地的

绑核只约束线程的运行 CPU。已经存在的页面、共享 page cache 和其他线程 first-touch 的内存仍可能位于远端。

### 19.2 Pinned Memory一定靠近GPU

Pinned 只描述页面被锁定。它位于哪个 Node，仍取决于分配、触页、驱动行为和内存策略。

### 19.3 整机CPU没满，线程就不会排队

某个 Node 或 cpuset 可以满载，而其他 CPU 空闲。线程是否被允许使用那些空闲 CPU 才是关键。

### 19.4 页面迁移次数大，迁移带宽就是主因

必须把页面数换算成字节和每秒带宽，再与业务数据量及内存带宽比较。

### 19.5 设置内存策略后变快，说明页面放对了

策略还可能改变自动 NUMA 扫描和线程放置。需要同时检查页面分布、线程分布和调度等待，并做方向相反的对照实验。

### 19.6 CPU利用率不对称就是调度器错误

不对称可能是调度器在 CPU 平衡与内存局部性之间作出的选择。真正的问题是这个选择是否符合业务的吞吐或延迟目标。

## 20. 可复用的分析框架

遇到 NUMA 性能问题时，可以按以下顺序思考：

```text
1. 拓扑
   CPU、内存、GPU、网卡分别属于哪个 Node？

2. 放置
   线程在哪里运行？关键物理页在哪里？谁 first-touch？

3. 路径
   数据是否跨 Node、跨 PCIe Root Complex？

4. 时间
   延迟花在 CPU 执行、runqueue 等待、阻塞还是 DMA？

5. 自动机制
   自动 NUMA 均衡是否迁移页面或影响线程调度？

6. 数量级
   页面迁移量、带宽和 CPU 时间能否解释实际退化？

7. 对照实验
   改变一个变量后，现象是否按预测方向变化？
```

最终要建立的证据链不是“NUMA 指标看起来异常”，而是：

```text
某种数据放置
    -> 产生特定远端访问模式
    -> 触发页面或线程迁移决策
    -> 改变每个 Node 的 CPU/内存负载
    -> 增加关键线程的执行或等待时间
    -> 导致业务指标退化
```

只有链路中的每一步都得到观测或实验支持，才能较有把握地确认根因。

## 21. 总结

NUMA 优化不是单独优化内存，也不是简单地把线程绑到 GPU 附近。它需要同时考虑：

- CPU 在哪里；
- 页面在哪里；
- 设备在哪里；
- 谁第一次触页；
- 哪些数据被线程共享；
- 调度器为什么移动线程；
- 自动 NUMA 均衡为什么移动页面；
- 关键线程是在执行，还是在 runqueue 中等待。

自动 NUMA 均衡导致不同 Node 的 CPU 利用率不一致，本质上是内核为了减少远端内存访问，让可迁移线程靠近共享热数据。当热数据集中在一个 Node 时，线程也会向该 Node 聚集。普通 CPU 负载均衡虽然试图分散任务，但还需要权衡内存局部性，因此不一定恢复完全对称的 CPU 利用率。

对于在线推理服务，需要同时优化完整数据路径：

```text
page cache -> CPU -> host memory -> pinned memory -> PCIe -> GPU
```

并确保延迟敏感线程拥有可预测的 CPU 时间。数据路径本地化与 CPU 资源隔离，缺少任何一项都可能留下性能问题。

## 22. 参考资料

- [Linux Kernel：NUMA Memory Policy](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html)
- [Linux man-pages：set_mempolicy(2)](https://man7.org/linux/man-pages/man2/set_mempolicy.2.html)
- [Linux Kernel：NUMA policy hit/miss statistics](https://www.kernel.org/doc/html/latest/admin-guide/numastat.html)
- [Linux Kernel：Kernel parameters](https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html)
