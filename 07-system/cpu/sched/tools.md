# CPU 调度观察工具

这篇笔记记录几个能直接观察 Linux 普通任务调度行为的入口：`nice`、CPU affinity、`/proc` 统计和内核源码位置。

## 1. 通过 nice 调整权重

Linux 普通任务通常走 `fair` 调度类。传统说法里它对应 CFS，Completely Fair Scheduler。

注意：从 Linux 6.6 开始，`fair` 调度类开始向 EEVDF 过渡，所以新内核源码里不仅能看到 CFS 的 `vruntime`，也能看到 EEVDF 的 `deadline`、`eligible` 等逻辑。

官方文档里对 CFS 的核心描述是：CFS 模拟一个“理想的多任务 CPU”。真实 CPU 一次只能运行一个任务，所以内核用 `vruntime` 记录任务在理想 CPU 上消耗了多少虚拟运行时间。

直觉上：

```text
实际运行时间越多，vruntime 越大
vruntime 越小，说明这个任务获得的 CPU 越少
调度器更倾向选择 vruntime 较小的任务
```

`nice` 不应该理解成 `nice * vruntime`。更准确地说，`nice` 会影响任务权重 `load.weight`，权重再影响真实运行时间折算成 `vruntime` 的速度。

```text
nice 越低 -> 权重越大 -> vruntime 增长越慢 -> 更容易获得更多 CPU
nice 越高 -> 权重越小 -> vruntime 增长越快 -> 获得的 CPU 更少
```

可以从 Linux 源码里这样找：

```text
kernel/sched/fair.c
```

关键结构和函数：

```text
struct sched_entity
    se->vruntime      // 虚拟运行时间
    se->load.weight   // 调度权重，受 nice 影响
    se->deadline      // 新内核 EEVDF 使用的虚拟截止时间

struct cfs_rq
    cfs_rq->tasks_timeline // fair 调度类的红黑树
    cfs_rq->avg_vruntime   // 平均虚拟运行时间
    cfs_rq->avg_load       // 平均权重

update_curr()
    更新当前任务的运行时间和 vruntime

calc_delta_fair()
    按权重把真实运行时间折算成虚拟运行时间

pick_next_task_fair()
    fair 调度类选择下一个任务的入口

DEFINE_SCHED_CLASS(fair)
    注册 fair 调度类的方法表
```

新版本 `fair.c` 中还能看到 EEVDF 的选择逻辑：

```text
eligible：任务是否被欠 CPU 时间
deadline：虚拟截止时间
__pick_eevdf()：从满足条件的任务中选择虚拟截止时间最早的任务
```

所以现在更严谨的说法是：

```text
Linux 普通任务属于 fair 调度类。
经典 CFS 用 vruntime 表达公平性。
新内核在 fair 调度类中引入 EEVDF，用 eligible + virtual deadline 改进选择逻辑。
```

## 2. 实验：两个 busy loop 竞争同一个 CPU

思路：

```text
1. fork 两个 CPU 密集型子进程
2. 把两个子进程绑到同一个 CPU
3. 一个保持 nice=0，另一个设置 nice=10
4. 从 /proc/<pid>/stat 读取 utime + stime
5. 比较同一段时间内两个进程拿到的 CPU time
```

```python
def busy_loop(name, nice_value):
    os.nice(nice_value)

    # 绑到同一个 CPU，避免它们跑到不同核上
    os.sched_setaffinity(0, {0})

    x = 0
    while True:
        x += 1
```

`bind_to_cpu0` 进行绑核，避免两个进程跑到不同 CPU 上；`setpriority` 设置 nice 值。

```cpp
void bind_to_cpu0() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::cerr << "sched_setaffinity failed: " << strerror(errno) << "\n";
        _exit(1);
    }
}

void busy_loop(int nice_value) {
    if (setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
        std::cerr << "setpriority failed: " << strerror(errno) << "\n";
        _exit(1);
    }

    bind_to_cpu0();

    volatile unsigned long long x = 0;
    while (true) {
        ++x;
    }
}
```

读取 CPU 时间时可以使用 `/proc/<pid>/stat`：

```text
第 14 列 utime：用户态 CPU ticks
第 15 列 stime：内核态 CPU ticks
cpu_time = (utime + stime) / sysconf(_SC_CLK_TCK)
```

运行方式：

```bash
g++ -O2 -std=c++17 cfs_demo.cpp -o cfs_demo
./cfs_demo
```

结果不要求精确固定，但通常 `nice=0` 的进程会比 `nice=10` 获得更多 CPU 时间。

## 3. 绑定核心

`cpu_set_t` 表示 CPU 核集合。

`CPU_ZERO` 清空集合，`CPU_SET` 把指定 CPU 加入集合，`sched_setaffinity` 设置当前进程可以运行在哪些 CPU 上。

```cpp
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(0, &set);
    CPU_SET(1, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        
    }
```

命令行也可以用 `taskset` 快速观察：

```bash
taskset -c 0 ./cfs_demo
taskset -pc <pid>
```

`taskset -c 0` 表示把进程限制在 CPU 0 上运行；`taskset -pc <pid>` 可以查看某个进程当前允许使用的 CPU 集合。

## 4. 常用命令

```bash
nice -n 10 ./a.out          # 以更低优先级启动普通任务
renice -n 10 -p <pid>       # 修改已有进程 nice
taskset -c 0 ./a.out        # 限制进程只在 CPU 0 运行
chrt -p <pid>               # 查看进程调度策略
ps -o pid,ni,pri,psr,stat,comm -p <pid>
```

`nice` 只影响普通任务在 `fair` 调度类中的权重；实时调度策略如 `SCHED_FIFO`、`SCHED_RR` 属于另一套调度类，不能和 CFS/EEVDF 的公平权重简单混在一起理解。
