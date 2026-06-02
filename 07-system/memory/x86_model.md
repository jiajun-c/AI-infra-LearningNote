# X86 内存模型

x86 的内存模型通常称为 TSO，Total Store Order。它比 ARM、RISC-V 这类弱内存模型更强，但不等于完全没有重排。

讨论内存模型时，需要先区分三件事：

```text
编译器重排：编译器在不改变单线程语义的前提下调整指令顺序
CPU 乱序执行：CPU 内部乱序发射/执行，但单线程提交结果保持架构语义
内存可见性重排：多核之间观察到的 load/store 顺序变化
```

x86 TSO 主要讨论的是第三种：一个核心上的内存操作，什么时候能被其他核心看到。

## TSO 基本规则

```cpp
Load  -> Load   不重排
Load  -> Store  不重排
Store -> Store  不重排
Store -> Load   可能看起来重排
```

更直观地说：

```text
同一个核心内，load/load、load/store、store/store 的顺序比较强。
主要例外是：前面的 store 可能还没有被其他核心看到，后面的 load 已经执行了。
```

这不是说 CPU 一定把两条指令真的交换了，而是从其他核心的观察角度看，效果像 `Store -> Load` 被重排。

## Store Buffer

`Store -> Load` 特殊的主要原因是 store buffer。

store buffer 是每个 CPU 核心私有的写缓冲队列。执行 store 时，核心可以先把写操作放进 store buffer，然后继续执行后续指令，不必立刻等待缓存一致性协议完成。

简化链路：

```text
store 指令
  -> 进入本核心 store buffer
  -> 等待拿到 cache line 独占权限
  -> 写入本核心 L1 cache
  -> 通过缓存一致性协议让其他核心可见
```

这样可以提高性能，但也带来一个现象：

```text
本核心认为自己已经写了
其他核心可能暂时还看不到这个写入
```

## 经典 Store Buffering 例子

初始状态：

```cpp
int x = 0;
int y = 0;
```

两个核心分别执行：

```cpp
// Core 0
x = 1;
r1 = y;

// Core 1
y = 1;
r2 = x;
```

在 x86 上，可能出现：

```text
r1 == 0 && r2 == 0
```

一种可能的过程：

```text
Core 0: x = 1 进入 Core 0 的 store buffer
Core 1: y = 1 进入 Core 1 的 store buffer

Core 0: 读取 y，此时 Core 1 的 y = 1 还没对 Core 0 可见，所以读到 0
Core 1: 读取 x，此时 Core 0 的 x = 1 还没对 Core 1 可见，所以读到 0
```

所以重点不是单线程语义被破坏，而是多核之间的可见性有延迟。

## Store Forwarding

同一个核心通常可以读到自己刚写过的值，这叫 store forwarding。

例如：

```cpp
x = 1;
r = x;
```

即使 `x = 1` 还在 store buffer 里，后面的 load 也可以直接从 store buffer 拿到值。

所以可以这样记：

```text
自己读自己的写：通常可以立刻看到
别人读你的写：可能要等 store buffer 刷出去之后才能看到
```

## Fence 和 lock 指令

如果需要约束内存操作顺序，可以使用内存屏障。

常见 x86 屏障：

```text
mfence：约束前后的 load/store，常作为 full memory fence 理解
lfence：主要约束 load
sfence：主要约束 store
```

例如：

```asm
mov [x], 1
mfence
mov rax, [y]
```

`mfence` 会要求前面的 store 不能继续只停在 store buffer 里，然后让后面的 load 提前越过去。

带 `lock` 前缀的原子指令通常也有很强的屏障效果：

```asm
lock xadd
lock cmpxchg
xchg
```

实际写 C/C++ 多线程代码时，一般不直接手写这些指令，而是使用语言层的同步原语。

## C++ atomic 的对应关系

`volatile` 不是线程同步工具。它主要影响编译器对该对象访问的优化，不提供完整的跨线程 happens-before 关系。

多线程共享数据应该使用：

```cpp
std::atomic
std::mutex
std::condition_variable
```

常见 atomic memory order：

```text
memory_order_relaxed：只保证单个原子对象的原子性，不保证跨变量顺序
memory_order_release：发布之前的写入
memory_order_acquire：获取 release 发布的写入
memory_order_seq_cst：最强顺序，提供更接近全局顺序的直觉
```

在 x86 上，硬件内存模型已经比较强，所以 acquire/release 往往不需要很重的硬件 fence；但编译器仍然需要按照 C++ 内存模型保留必要约束。写并发代码时应该按 C++ 语义理解，而不是依赖某个平台的偶然实现。

## 和 ARM/RISC-V 的区别

x86 TSO 比 ARM、RISC-V 常见模型更强：

```text
x86：大多数普通 load/store 顺序天然较强，主要关注 Store -> Load
ARM/RISC-V：更多顺序可能需要 acquire/release 或 fence 明确约束
```

因此一些在 x86 上“看起来能跑”的无锁代码，移植到 ARM/RISC-V 后可能暴露问题。

## 小结

```text
x86 TSO 的核心记忆：
1. 单线程看起来按程序顺序执行
2. CPU 内部可以乱序执行，但提交结果保持架构语义
3. 多核下大多数内存顺序较强
4. 主要例外是 Store -> Load 可能看起来重排
5. 这个例外主要来自每个核心私有的 store buffer
6. 跨线程同步应使用 atomic、mutex 或 fence，不要依赖 volatile
```
