# x86 cache 架构

x86 的 cache 通常是多级结构。越靠近核心越快、越小，越靠近内存越慢、越大。

简化结构：

```text
Core 0
  L1 I-cache
  L1 D-cache
  L2 cache
        \
Core 1    \
  L1 I-cache \
  L1 D-cache  \
  L2 cache     -> Shared L3 cache -> Memory Controller -> DRAM
```

## 1. L1 cache

L1 cache 是每个 core 私有的一级缓存，离执行单元最近，延迟最低。

它通常分成两部分：

```text
L1 I-cache：Instruction Cache，缓存指令
L1 D-cache：Data Cache，缓存数据
```

常见特点：

```text
容量小：常见 32KB I-cache + 32KB D-cache
延迟低：几 cycles
cache line：x86 常见 64 bytes
```

## 2. L2 cache

L2 cache 通常也是每个 core 私有。它比 L1 更大，但延迟更高。

作用：

```text
承接 L1 miss
缓存当前 core 最近使用但放不进 L1 的数据
减少访问共享 L3 的压力
```

## 3. L3 cache

L3 cache 通常是多个 core 共享的最后一级缓存，也叫 LLC，Last Level Cache。

作用：

```text
承接 L2 miss
减少访问 DRAM
帮助多核心之间共享数据和维护缓存一致性
```

典型延迟层级：

```text
L1：几 cycles
L2：十几 cycles
L3：几十 cycles
DRAM：上百 cycles
```

具体数值会随着 CPU 架构、频率、内存配置、NUMA、预取命中情况变化。

## 4. 测试 cache 延迟

可以用目录里的 `cache_latency.cpp` 做一个简单 pointer chasing benchmark。

编译：

```bash
g++ -O2 -std=c++17 cache_latency.cpp -o cache_latency
```

运行：

```bash
./cache_latency
```

这个程序会构造不同大小的随机链表，每个节点占一个 64B cache line，然后做依赖式访问：

```cpp
index = nodes[index].next;
```

因为下一次访问地址依赖上一次 load 的结果，CPU 很难把多个访问并行起来；随机顺序也会削弱硬件预取，所以测出来的 `cycles/load` 更接近不同层级 cache miss/hit 的延迟。

结果通常会看到几个台阶：

```text
工作集小于 L1：接近 L1 延迟
超过 L1、小于 L2：接近 L2 延迟
超过 L2、小于 L3：接近 L3 延迟
超过 L3：更多访问落到 DRAM
```

注意：这不是严格的硬件规格测试，只是用来观察延迟层级。真实结果会受到 CPU 型号、Turbo、绑定核心、TLB、页大小、NUMA 和后台负载影响。
