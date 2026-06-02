# IPC

## 1. 管道

管道本质是内核维护的一片环形缓冲区，一端写入，另外一端读出

管道的特点
- 半双工：数据只能单向流动
- 字节流：无格式，无边界
- 亲缘关系：匿名管道
- 生命周期：随进程销毁
- 阻塞行为：管道满时 write 阻塞，管道空时 read 阻塞

## 代码验证

示例代码：[pipe_features_demo.c](./pipe_features_demo.c)

```bash
make
./pipe_features_demo
```

或直接：

```bash
make run
```

这个程序会依次验证：

- 半双工：`pipe()` 返回读端和写端，对读端 `write`、对写端 `read` 会失败。
- 字节流：连续两次 `write("ABC")`、`write("DEF")` 后，`read(4)` 可以读到跨越写入边界的 `"ABCD"`。
- 亲缘关系：匿名管道没有路径名，子进程通过 `fork()` 继承父进程的 fd 后才能读到父进程写入的数据。
- 生命周期：所有写端关闭后，读端 `read()` 返回 `0`，表示 EOF。
- 阻塞行为：空管道上的 `read()` 会等到有数据；满管道上的 `write()` 会等到读端释放空间。

## 2. 共享内存

共享内存，shared memory，是一种让多个进程把同一片物理内存映射到各自虚拟地址空间的 IPC 方式。

普通进程之间的地址空间是隔离的：

```text
process A virtual memory -> physical memory A
process B virtual memory -> physical memory B
```

共享内存会让两个进程的不同虚拟地址指向同一片物理页：

```text
process A virtual addr 0x7f... ----\
                                    -> same physical pages
process B virtual addr 0x7e... ----/
```

这样进程 A 写入共享内存后，进程 B 可以直接从自己的地址空间读到，不需要通过内核在两个进程之间拷贝数据。

### 为什么共享内存快

管道、socket、消息队列这类 IPC 通常有内核参与数据传递：

```text
process A user buffer -> kernel buffer -> process B user buffer
```

共享内存创建和映射时需要内核参与，但数据读写本身就是普通内存访问：

```text
process A writes shared page
process B reads same shared page
```

所以共享内存适合：

- 大块数据传输
- 高频生产者消费者队列
- 图像、音频、日志 buffer
- 多进程共享状态

### 共享内存只解决“数据共享”

共享内存本身不解决同步问题。

例如进程 A 正在写：

```text
message = "hello"
ready = 1
```

进程 B 什么时候读？读到一半怎么办？多个进程同时写怎么办？

这些都需要额外同步机制，例如：

- semaphore：信号量
- mutex：进程间互斥锁，通常放在共享内存里，并设置 `PTHREAD_PROCESS_SHARED`
- futex：Linux 底层同步原语
- eventfd：用于通知另一个进程
- pipe/socket：只用来做唤醒通知，数据走共享内存

共享内存的典型组合是：

```text
shared memory: 传输数据
semaphore/mutex/eventfd: 通知和同步
```

## 3. POSIX 共享内存

POSIX 共享内存常用 API：

```c
shm_open();
ftruncate();
mmap();
munmap();
close();
shm_unlink();
```

基本流程：

```text
1. shm_open 创建或打开共享内存对象
2. ftruncate 设置共享内存大小
3. mmap 把共享内存映射到当前进程地址空间
4. 像普通内存一样读写
5. munmap 解除映射
6. close 关闭 fd
7. shm_unlink 删除共享内存名字
```

示例伪代码：

```c
int fd = shm_open("/demo_shm", O_CREAT | O_RDWR, 0666);
ftruncate(fd, 4096);

char *p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

strcpy(p, "hello from process A");

munmap(p, 4096);
close(fd);
shm_unlink("/demo_shm");
```

另一个进程用同样的名字打开：

```c
int fd = shm_open("/demo_shm", O_RDWR, 0666);
char *p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

printf("%s\n", p);
```

注意：

- POSIX 共享内存名字通常以 `/` 开头，比如 `/demo_shm`
- `ftruncate` 决定共享内存对象大小
- `MAP_SHARED` 表示映射内容对其他进程可见
- `shm_unlink` 删除的是名字；已经映射的进程仍可继续使用，直到解除映射

## 4. mmap 文件共享

除了 `shm_open`，也可以把普通文件映射为共享内存。

```text
open file -> ftruncate -> mmap(MAP_SHARED)
```

这种方式的特点：

- 多进程映射同一个文件
- 修改可以回写到文件
- 适合需要持久化的数据

伪代码：

```c
int fd = open("data.bin", O_CREAT | O_RDWR, 0666);
ftruncate(fd, 4096);

char *p = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
p[0] = 'A';

msync(p, 4096, MS_SYNC);
munmap(p, 4096);
close(fd);
```

`shm_open` 更像“以内存为主的共享区域”，普通文件 `mmap` 更像“把文件当内存访问”。

## 5. System V 共享内存

System V 共享内存是更老的一套接口。

常用 API：

```c
shmget();
shmat();
shmdt();
shmctl();
```

基本流程：

```text
1. shmget 创建或获取共享内存段
2. shmat attach 到当前进程地址空间
3. 像普通内存一样读写
4. shmdt detach
5. shmctl(..., IPC_RMID, ...) 标记删除
```

对比 POSIX：

```text
POSIX:    shm_open -> fd -> mmap
System V: shmget   -> shmid -> shmat
```

现代 Linux 程序里，POSIX 共享内存和 `mmap` 文件映射更常见，接口也更接近普通 fd 模型。

## 6. 生命周期

共享内存有两层生命周期：

- 名字或 ID 的生命周期
- 实际映射和物理页的生命周期

以 POSIX 共享内存为例：

```text
shm_unlink("/demo_shm")
```

只是删除名字，防止新的进程通过这个名字打开它。已经打开 fd 或已经 `mmap` 的进程仍然能继续使用。

真正释放通常要等：

```text
没有名字 + 没有打开 fd + 没有进程映射
```

这个行为和文件的 `unlink` 很像：删除目录项不等于立刻释放已打开文件。

## 7. 一致性问题

共享内存常见 bug 不是“读不到”，而是“读到了不一致的数据”。

例如共享结构：

```c
struct Message {
    int ready;
    size_t len;
    char data[1024];
};
```

错误顺序可能是：

```text
writer sets ready = 1
writer copies data
reader sees ready = 1
reader reads half-written data
```

正确做法通常是：

```text
writer lock
writer writes data and len
writer sets ready
writer unlock or post semaphore

reader wait semaphore or lock
reader checks ready
reader reads data
reader unlock
```

如果使用无锁结构，还要考虑：

- 原子变量
- memory ordering
- cache line false sharing
- 单生产者单消费者还是多生产者多消费者

## 8. 和其他 IPC 的对比

| IPC 方式 | 数据拷贝 | 是否需要同步 | 适合场景 |
| --- | --- | --- | --- |
| 管道 | 经过内核 buffer | 内核提供阻塞语义 | 父子进程、字节流 |
| Unix domain socket | 经过内核 | 内核提供队列语义 | 本机进程间通用通信 |
| 消息队列 | 经过内核 | 内核提供消息边界 | 小消息、异步通知 |
| 共享内存 | 数据读写不需要内核拷贝 | 需要自己处理同步 | 大数据、高频通信 |

共享内存的优势是快，代价是复杂：你需要自己设计数据结构、边界、同步和生命周期。

## 9. 小结

共享内存通信可以理解为：

```text
内核负责建立共享映射，进程负责直接读写，同步机制负责保证读写顺序。
```

最常见的工程形态：

```text
shm_open/mmap 创建共享 buffer
semaphore/eventfd 通知对方
mutex/atomic 保护共享状态
```

抓住三件事就不容易乱：

- 数据在哪里：共享内存区域
- 谁能看到：映射了同一对象的进程
- 何时能读写：由锁、信号量、事件通知或原子变量决定
