# Semaphore

信号量，semaphore，是一种用于同步的 IPC/并发原语。它本质上是一个带等待队列的计数器。

注意：信号量 semaphore 和信号 signal 不是一回事。

- signal：异步通知机制，例如 `SIGINT`、`SIGKILL`
- semaphore：同步机制，用来控制并发访问、进程/线程等待和唤醒

## 1. 核心机制

信号量内部可以抽象成：

```text
struct semaphore {
    int count;
    wait_queue queue;
};
```

它有两个核心操作：

```text
P / wait / down:
    count--
    如果 count < 0，当前线程或进程进入等待队列并阻塞

V / post / up:
    count++
    如果有人在等待队列里，唤醒一个等待者
```

也可以用更直观的方式理解：

```text
wait:
    如果资源可用，拿走一个资源
    如果资源不可用，睡眠等待

post:
    归还一个资源
    如果有人等待，唤醒一个
```

信号量的关键点：

- `count` 表示可用资源数量
- `wait` 可能阻塞
- `post` 通常不会阻塞
- 阻塞和唤醒由内核或线程库管理

## 2. 二值信号量和计数信号量

### 二值信号量

二值信号量的值只有 `0/1`，常用来做互斥。

```text
count = 1: 临界区空闲
count = 0: 临界区被占用
```

使用方式：

```text
wait(sem)
critical section
post(sem)
```

这和 mutex 很像，但语义略有不同：

- mutex 强调“锁的拥有者”，通常谁加锁谁解锁
- semaphore 更像“资源计数”，一个线程 wait，另一个线程 post 也可以成立

### 计数信号量

计数信号量的值可以大于 1，用来表示多个同类资源。

例如有 4 个 buffer：

```text
sem = 4
```

每个 worker 使用一个 buffer 前：

```text
wait(sem)
use buffer
post(sem)
```

最多只允许 4 个 worker 同时进入。

## 3. 生产者消费者模型

信号量最经典的用法是生产者消费者。

假设有一个大小为 `N` 的环形队列：

```text
empty = N    // 空槽数量
full  = 0    // 已有数据数量
mutex = 1    // 保护队列结构
```

生产者：

```text
wait(empty)      // 等待空槽
wait(mutex)      // 锁住队列
push item
post(mutex)      // 解锁队列
post(full)       // 通知有数据
```

消费者：

```text
wait(full)       // 等待数据
wait(mutex)      // 锁住队列
pop item
post(mutex)      // 解锁队列
post(empty)      // 归还空槽
```

这里三个信号量分别解决不同问题：

- `empty`：控制生产者不能把队列写爆
- `full`：控制消费者不能从空队列读取
- `mutex`：保护队列内部状态，避免并发修改

## 4. 为什么 wait/post 必须是原子的

如果 `wait` 不是原子操作，就会出现竞争。

错误情况：

```text
sem = 1

thread A reads sem = 1
thread B reads sem = 1
thread A sets sem = 0 and enters
thread B sets sem = 0 and enters
```

结果两个线程都进入临界区，互斥失败。

所以信号量的 `wait/post` 必须依赖原子指令、内核锁或者 futex 这类机制，保证检查计数和修改计数不可被打断。

## 5. 和共享内存的关系

共享内存只负责让多个进程看到同一份数据，不负责读写顺序。

例如：

```text
process A writes shared memory
process B reads shared memory
```

如果没有同步，B 可能读到：

- A 还没写的数据
- A 写了一半的数据
- 多个 writer 交错后的错误数据

因此共享内存通常要配合信号量：

```text
shared memory: 存放数据
semaphore: 通知数据是否可读/可写
```

一个简单的单消息模型：

```text
empty = 1
full  = 0
```

写进程：

```text
wait(empty)
write shared memory
post(full)
```

读进程：

```text
wait(full)
read shared memory
post(empty)
```

这样可以保证：

- 写之前，buffer 是空的
- 读之前，buffer 里有完整数据
- 读完后，写方才能覆盖旧数据

## 6. POSIX 信号量

POSIX 信号量有两种常见形式。

### 命名信号量

命名信号量可以被无亲缘关系的进程通过名字打开。

常用 API：

```c
sem_open();
sem_wait();
sem_post();
sem_close();
sem_unlink();
```

典型流程：

```c
sem_t *sem = sem_open("/demo_sem", O_CREAT, 0666, 1);

sem_wait(sem);
// critical section
sem_post(sem);

sem_close(sem);
sem_unlink("/demo_sem");
```

特点：

- 名字通常以 `/` 开头
- 可以用于无亲缘关系进程
- `sem_unlink` 删除名字，不一定立刻销毁已经打开的信号量

### 匿名信号量

匿名信号量没有名字，通常放在共享内存里。

常用 API：

```c
sem_init();
sem_wait();
sem_post();
sem_destroy();
```

线程内共享：

```c
sem_init(&sem, 0, initial_value);
```

进程间共享：

```c
sem_init(&sem, 1, initial_value);
```

第二个参数 `pshared` 的含义：

- `0`：线程间共享
- 非 `0`：进程间共享，信号量对象需要放在共享内存里

## 7. System V 信号量

System V 信号量是更老的一套 IPC 接口。

常用 API：

```c
semget();
semop();
semctl();
```

特点：

- 可以一次创建一组信号量
- 接口比 POSIX 更复杂
- 常用于老系统或历史代码

对比：

```text
POSIX semaphore: sem_open / sem_init / sem_wait / sem_post
System V:        semget / semop / semctl
```

现代代码里，POSIX 信号量更直观。

## 8. 信号量和 mutex/condition variable

### semaphore vs mutex

```text
mutex:
    主要保护临界区
    强调 owner
    一般谁 lock 谁 unlock

semaphore:
    主要表示资源数量或事件次数
    不强调 owner
    一个线程 wait，另一个线程 post 很常见
```

### semaphore vs condition variable

condition variable 通常和 mutex 配合，用来等待某个条件变为真。

```text
while (!condition) {
    pthread_cond_wait(&cond, &mutex);
}
```

condition variable 本身不保存“通知次数”。如果先 signal，后 wait，通知可能丢失。

semaphore 的 `count` 可以保存通知次数：

```text
post before wait -> count 增加
later wait -> 可以直接通过
```

所以 semaphore 很适合表达“有 N 个事件/资源可消费”。

## 9. 常见问题

### 死锁

如果进程 wait 后忘记 post，其他等待者可能永远阻塞。

```text
wait(sem)
return error without post
```

临界区里要特别注意异常路径。

### 初始值错误

初始值决定资源数量。

```text
mutex semaphore: 初始值通常是 1
empty slots:     初始值通常是 buffer size
full slots:      初始值通常是 0
```

初始值设错，程序可能一开始就阻塞，或者互斥失效。

### 忘记进程间共享条件

如果匿名信号量要跨进程使用：

- 信号量对象必须在共享内存里
- `sem_init` 的 `pshared` 参数必须非 0

否则 fork 后每个进程可能操作自己的副本，同步不会生效。

### 被信号中断

`sem_wait` 可能被 signal 中断并返回 `-1`，`errno == EINTR`。

健壮代码通常需要重试：

```c
while (sem_wait(&sem) == -1 && errno == EINTR) {
}
```

## 10. 小结

信号量可以用一句话理解：

```text
信号量 = 可用资源计数 + 等待队列 + 原子的 wait/post
```

它解决的是“什么时候可以继续执行”的问题：

- `count > 0`：可以继续，消耗一个资源
- `count == 0`：不能继续，阻塞等待
- `post`：释放资源或发送通知，唤醒等待者

在 IPC 里，信号量常和共享内存一起使用：

```text
共享内存负责放数据，信号量负责控制读写时机。
```
