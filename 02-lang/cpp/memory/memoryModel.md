# 内存模型

## 1. X86内存模型

x86采用了一个较强的内存模式，叫做TSO(Total Store Order)

```cpp
CPU核 A CPU 核 B
+--------+ +--------+
| Store  | |  Load  |
| Buffer |              |        |
+----+---+              |        |
     |                  +--------+
     v+----------------------+
| L1 /共享 L3 |
+----------------------+
```

关键特性

- Store Buffer顺序保证：每个CPU核有一个store buffer，store先进入该buffer，再异步写入L1/共享缓存，同一线程的Store顺序对外可见，
- 在TSO中使用了FIFO的Store Buffer，假设线程A有两个写入，线程B在看到后一个写入的时候保证前一个写入被看见
- 允许LoadLoad，LoadStore，StoreStore重排
- 原子指令：有LOCK XADD

## 2. ARM内存模型

ARM内存模式其实是一个弱内存模型，CPU可以激进地执行Load/Store，只有遇到显式barrier才会强制刷出，结构示意

```cpp
CPU 核 A                        CPU 核 B
+---------------------+         +---------------------+
| Load Queue |         | Load Queue          |
| Store Buffer |         | Store Buffer        |
| Local Bypass / FU |         | Local Bypass / FU   |
+----------+----------+         +----------+----------+
           \ /
            +------ Shared Bus / Coherence ----+
                       |
                +------+------+
                |  L2 / 主存 |
                +-------------+
```

- 几乎所有的重排都被允许
- 同一线程的程序顺序对其他线程是不可见
- Barrier指令
  - DMB(Data Memory Barrier)：保证前后访存的顺序
  - DSB(Data Synchronization Barrier):更严格，等待所有的访存完成
  - ISB(Instruction Synchronization Barrier)：刷流水线

## 3. GPU内存模型

很弱(按需同步)，默认是没有一致性的，因为GPU的并行度极高，核心数量极多

