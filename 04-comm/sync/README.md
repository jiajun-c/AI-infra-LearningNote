# 跨卡同步机制

分布式训练中的同步本质上是让多个 rank（进程/GPU）在执行时序上对齐。`torch.distributed` 提供了从粗粒度 barrier 到精细 P2P 同步的多层接口。

## 1. 基于dist的同步

最基础的同步原语。所有调用 `barrier()` 的 rank 会被阻塞，直到所有 rank 都到达这一行代码。适合用于训练开始前的初始化对齐、benchmark 计时同步等场景。

```python
import torch.distributed as dist

# 初始化（每个进程都会执行）
dist.init_process_group(backend="nccl", init_method="env://")

dist.barrier()
```

dist.barrier 其底层的实现还是一个NCCL的Allreduce，如下所示

```cpp
auto work = allreduce_impl(barrierTensor, "nccl:all_reduce_barrier", arOpts);
// ↓
// allreduce_impl 内部调用 collective()，最终调用:
return ncclAllReduce(
    input.data_ptr(),      // dummy tensor 的地址
    output.data_ptr(),
    input.numel(),         // = 1 (只有一个元素!)
    ncclDataType,          // ncclFloat32
    ncclReduceOp,          // SUM
    comm,                  // NCCL communicator
    stream.stream()        // CUDA stream
);
```

## 2. 基于cuda device API的同步

`cudaStreamWriteValue32`/`cudaStreamWaitValue32` 是 CUDA 提供的**流序信号量（stream-ordered semaphore）**原语。与 `dist.barrier()` 不同，这种同步**完全在 GPU 上完成，不阻塞 CPU**——一个 GPU 的 stream 直接写一个值到内存，另一个 GPU 的 stream 轮询等待该值变化。

### 2.1 核心 API

```cpp
// GPU stream 向内存地址 ptr 写入 32-bit 值 value
__host__ cudaError_t cudaStreamWriteValue32(
    cudaStream_t stream, void* ptr, cuuint32_t value, unsigned int flags);

// GPU stream 阻塞等待，直到 *ptr 满足条件（与 value 比较）
__host__ cudaError_t cudaStreamWaitValue32(
    cudaStream_t stream, const void* ptr, cuuint32_t value, unsigned int flags);
```

|flags (WaitValue)|含义|
|---|---|
|`cudaStreamWaitValueEq` (0)|等待 `*ptr == value`|
|`cudaStreamWaitValueGte`|等待 `*ptr >= value`|
|`cudaStreamWaitValueNorMask`|等待 `(*ptr & value) == 0`|
|`cudaStreamWaitValueAndMask`|等待 `(*ptr & value) != 0`|
|`cudaStreamWaitValueFlush`|每次检查前刷新远端写，避免读到旧 cache 值|

### 2.2 跨设备内存：为什么需要 Pinned Host Memory

GPU 之间不能直接访问对方的 device memory。跨 GPU 的 WriteValue/WaitValue 需要一块**所有 GPU 都能访问**的内存：

```text
        ┌─────────────┐         ┌─────────────┐
        │   GPU 0     │         │   GPU 1     │
        │  stream_0   │         │  stream_1   │
        │    │        │         │    │        │
        │    │ Write  │         │    │ Wait   │
        │    ▼        │         │    ▼        │
        └────┬────────┘         └────┬────────┘
             │                       │
             │    PCIe / NVLink      │
             └───────────┬───────────┘
                         ▼
              ┌─────────────────────┐
              │  Pinned Host Memory │  ← cudaHostAlloc(cudaHostAllocPortable)
              │  volatile uint32_t  │     CPU/所有 GPU 都可访问
              │  signal = 0 → 1    │
              └─────────────────────┘
```

### 2.3 完整示例：GPU 0 通知 GPU 1 开始工作

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define check(call)                                                  \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(1);                                                 \
        }                                                            \
    } while (0)

__global__ void heavy_compute_kernel(float* data, int n, int dev_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];
        for (int i = 0; i < 1000; i++) {
            val = sinf(cosf(val));
        }
        data[tid] = val;
    }
    if (tid == 0) {
        printf("[GPU %d] compute done\n", dev_id);
    }
}

int main() {
    const int N = 1 << 20;
    const float one_mb = (float)(N * sizeof(float)) / (1024 * 1024);

    // ========== Step 1: 分配所有 GPU 都能看到的信号内存 ==========
    // cudaHostAllocPortable：允许所有 GPU 访问这块 pinned memory
    volatile uint32_t* signal;
    check(cudaHostAlloc((void**)&signal, sizeof(uint32_t),
                         cudaHostAllocPortable));
    *signal = 0;

    // ========== Step 2: 每个 GPU 分配自己的工作内存 ==========
    int dev_count;
    check(cudaGetDeviceCount(&dev_count));
    if (dev_count < 2) {
        fprintf(stderr, "Need at least 2 GPUs\n");
        return 1;
    }

    cudaStream_t stream[2];
    float *d_data[2];

    for (int i = 0; i < 2; i++) {
        check(cudaSetDevice(i));
        check(cudaStreamCreate(&stream[i]));
        check(cudaMalloc(&d_data[i], N * sizeof(float)));
    }

    // ========== Step 3: 构建 GPU 间的同步链 ==========
    // GPU 0: 做计算 → 写 signal = 1 → 通知 GPU 1
    check(cudaSetDevice(0));
    heavy_compute_kernel<<<256, 256, 0, stream[0]>>>(d_data[0], N, 0);
    // ★ 计算完成后，在 GPU 0 的 stream 上写 signal=1
    check(cudaStreamWriteValue32(stream[0], (void*)signal, 1,
                                  cudaStreamWriteValueDefault));

    // GPU 1: 等待 signal == 1 → 再做计算
    check(cudaSetDevice(1));
    // ★ GPU 1 的 stream 在此阻塞，直到 signal 变成 1
    check(cudaStreamWaitValue32(stream[1], (const void*)signal, 1,
                                 cudaStreamWaitValueEq));
    heavy_compute_kernel<<<256, 256, 0, stream[1]>>>(d_data[1], N, 1);

    // ========== Step 4: CPU 等待所有 GPU 完成 ==========
    for (int i = 0; i < 2; i++) {
        check(cudaSetDevice(i));
        check(cudaStreamSynchronize(stream[i]));
    }

    printf("All done! GPU 1 waited for GPU 0's signal in pure GPU path.\n");

    // cleanup
    for (int i = 0; i < 2; i++) {
        check(cudaSetDevice(i));
        check(cudaStreamDestroy(stream[i]));
        check(cudaFree(d_data[i]));
    }
    check(cudaFreeHost((void*)signal));
    return 0;
}
```

编译运行：

```shell
nvcc -o sync_demo sync_demo.cu
./sync_demo
```

### 2.4 时间线图解

```text
         T0      T1      T2      T3      T4
GPU 0:  [────── compute ──────────]─►
                                       │ WriteValue(signal=1)
                                       ▼
         ┌─────────────────────────────────────────┐
         │  signal:  0  ─────────────────→  1      │
         └─────────────────────────────────────────┘
                                                   │
GPU 1:  ───►  [WaitValue: signal==1 ?] ────────────┼── [────── compute ──────]─►
           ▲                                        │
           │ GPU 线程不停 poll signal              │
           │ 每次检查前 flush L2 cache              │
           │ 直到 signal == 1 才往下走              │
```

**关键差异 vs `dist.barrier()`**：

|维度|`dist.barrier()` (NCCL)|`cudaStreamWriteValue` + `WaitValue`|
|---|---|---|
|阻塞位置|CPU 线程 + GPU stream|**仅 GPU stream**，CPU 不阻塞|
|同步粒度|所有 rank 全体同步|任意 GPU 之间的点对点|
|通信量|allreduce 至少一轮 ring|一次 32-bit 内存写入|
|延迟|受 ring/tree 跳数影响|受 PCIe/NVLink 访存延迟影响|
|适用场景|分布式训练梯度同步|GPU 间流水线、持久化 kernel 通知|

### 2.5 多 GPU 生产者-消费者流水线

一个更实际的场景：**GPU 0 产生中间结果，GPU 1 消费这些结果**，无需 CPU 介入：

```cpp
// 多槽位的生产者-消费者信号量
volatile uint32_t* ready_flag;   // GPU 0 写，GPU 1 等
volatile uint32_t* done_flag;    // GPU 1 写，GPU 0 等

check(cudaHostAlloc((void**)&ready_flag, sizeof(uint32_t),
                     cudaHostAllocPortable));
check(cudaHostAlloc((void**)&done_flag, sizeof(uint32_t),
                     cudaHostAllocPortable));
*ready_flag = 0;
*done_flag = 0;

const int STAGES = 4;
for (int step = 0; step < STAGES; step++) {
    // GPU 0: compute stage → signal ready
    check(cudaSetDevice(0));
    produce_kernel<<<grid, block, 0, stream[0]>>>(output_buf, step);
    check(cudaStreamWriteValue32(stream[0], (void*)ready_flag, step + 1,
                                  cudaStreamWriteValueDefault));

    // GPU 1: wait ready → consume → signal done
    check(cudaSetDevice(1));
    check(cudaStreamWaitValue32(stream[1], (const void*)ready_flag, step + 1,
                                 cudaStreamWaitValueEq));
    consume_kernel<<<grid, block, 0, stream[1]>>>(output_buf, step);
    check(cudaStreamWriteValue32(stream[1], (void*)done_flag, step + 1,
                                  cudaStreamWriteValueDefault));

    // GPU 0: wait done before reusing buffer
    check(cudaSetDevice(0));
    check(cudaStreamWaitValue32(stream[0], (const void*)done_flag, step + 1,
                                 cudaStreamWaitValueEq));
}
```

这个模式里 **CPU 只负责提交 kernel，不参与任何同步**——生产者 GPU 做完就写 flag，消费者 GPU 等到 flag 就开始消费，形成了纯 GPU 驱动的流水线。

## 3. 基于IPC的同步

前两节都在**同一进程**内同步。当两个独立的进程（各自的地址空间）需要协调 GPU 流时，需要 CUDA IPC 机制。核心是两个 API：

- **IPC Event**：进程 A 记录一个 event，进程 B 的 stream 等待这个 event，实现**跨进程的 GPU 流同步**
- **IPC Memory**：共享 device memory 指针，配合 `cudaStreamWriteValue32/WaitValue32` 实现跨进程信号量

### 3.1 IPC Event：跨进程 GPU 流同步

IPC Event 是 CUDA 专门为跨进程 GPU 同步设计的机制。流程如下：

```text
进程 A (GPU 0)                    共享通道              进程 B (GPU 0)
─────────────                    ────────              ─────────────
kernel_A<<<>>>                   (socket / shm /      recv(handle_bytes)
cudaEventRecord(eventA, streamA)  cudaMemcpy to host)
cudaEventSynchronize(eventA) ──── handle ──────────→  cudaIpcOpenEventHandle(&ipcEvent, handle)
                        发送 IPC handle (64 bytes)    cudaStreamWaitEvent(streamB, ipcEvent)
                                                      kernel_B<<<>>>
                                                      ↑ kernel_B 只在 kernel_A 完成后才执行
```

**为什么要 `cudaEventSynchronize` 之后再发 handle？**
IPC event handle 只有在 event **被创建且至少 record 过一次**后才有效。先同步确保 GPU 上的 record 操作已提交，再导出 handle。

#### 完整代码示例（通过 POSIX 共享内存传递 handle）

```cpp
// ---------- 进程 A: 生产者 ----------
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

__global__ void produce_kernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) data[tid] = sinf(data[tid]);
}

void producer() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 分配 GPU 内存并执行计算
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    produce_kernel<<<1, 256, 0, stream>>>(d_data, 1024);

    // ★ 创建并记录 event，然后导出 IPC handle
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);

    // 必须在 IPC export 前确保 event 已提交到 GPU
    cudaEventSynchronize(event);

    // 导出 IPC handle（64 字节固定大小）
    cudaIpcEventHandle_t handle;
    cudaIpcGetEventHandle(&handle, event);

    // 通过 POSIX 共享内存发送到进程 B
    int fd = shm_open("/ipc_sync_shm", O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(cudaIpcEventHandle_t));
    void* shm = mmap(NULL, sizeof(cudaIpcEventHandle_t),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    memcpy(shm, &handle, sizeof(cudaIpcEventHandle_t));

    printf("Producer: IPC handle sent, waiting for consumer readiness...\n");

    // 消费者完成前，不能销毁 event（否则 IPC handle 失效）
    // 简单做法：等消费者发回确认信号...
    munmap(shm, sizeof(cudaIpcEventHandle_t));
    close(fd);

    cudaStreamSynchronize(stream);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
}

// ---------- 进程 B: 消费者 ----------
__global__ void consume_kernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) data[tid] = cosf(data[tid]);
}

void consumer() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 从共享内存读取 IPC handle
    int fd = shm_open("/ipc_sync_shm", O_RDONLY, 0666);
    cudaIpcEventHandle_t handle;
    pread(fd, &handle, sizeof(cudaIpcEventHandle_t), 0);

    // ★ 使用 IPC handle 创建一个本地可等待的 event
    cudaEvent_t ipcEvent;
    cudaIpcOpenEventHandle(&ipcEvent, handle);

    // ★ 让当前 stream 等待远程 event ——
    //    进程 B 的 GPU stream 会阻塞，直到进程 A 的 event 完成
    cudaStreamWaitEvent(stream, ipcEvent);

    // 这个 kernel 只在进程 A 的 produce_kernel 完成后才执行
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    consume_kernel<<<1, 256, 0, stream>>>(d_data, 1024);

    printf("Consumer: kernel launched, waiting on IPC event...\n");
    cudaStreamSynchronize(stream);
    printf("Consumer: done!\n");

    cudaEventDestroy(ipcEvent);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    munmap(&handle, sizeof(cudaIpcEventHandle_t));
    close(fd);
    shm_unlink("/ipc_sync_shm");
}
```

### 3.2 时间线对比

```text
             T0      T1      T2      T3      T4      T5
进程 A ── [produce_kernel] ──►
             record(event) ─┐
                            │ IPC handle via shm
进程 B ────── 读取 handle ──┘
             openIpcEvent()
             cudaStreamWaitEvent(ipcEvent)
             ───── 阻塞等待 event ─────► [consume_kernel] ──►
                            ↑                         ↑
                  进程 A event 完成             进程 B 的 stream 才被唤醒
```

关键点：**CPU 完全不参与 GPU 同步**。进程 B 的 CPU 线程提交完 `cudaStreamWaitEvent` 和 `consume_kernel` 后就可以立即返回——所有等待都在 GPU 上完成。

### 3.3 IPC Memory + Stream-Ordered Value：更灵活的跨进程信号量

把第 2 节的 `cudaStreamWriteValue32/WaitValue32` 和 IPC 结合起来：用 `cudaIpcGetMemHandle/cudaIpcOpenMemHandle` 共享一块 device memory 作为信号量，实现**细粒度的跨进程 GPU-GPU 同步**。

```cpp
// ---------- 共享的信号量结构 ----------
// 在两个进程间共享
typedef struct {
    volatile uint32_t ready;   // 生产者写，消费者等
    volatile uint32_t done;    // 消费者写，生产者等
} SyncFlag;

// ---------- 进程 A: 共享自己的 device memory ----------
void producer_with_ipc_mem() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 分配信号量内存
    SyncFlag* d_flag;
    cudaMalloc(&d_flag, sizeof(SyncFlag));
    cudaMemset(d_flag, 0, sizeof(SyncFlag));

    // 导出 IPC 内存句柄
    cudaIpcMemHandle_t memHandle;
    cudaIpcGetMemHandle(&memHandle, (void*)d_flag);

    // 通过共享内存发送 memHandle + 设备指针偏移
    // ... (send memHandle via shm/socket) ...

    // compute → signal ready
    produce_kernel<<<1, 256, 0, stream>>>(/*...*/);
    cudaStreamWriteValue32(stream, &d_flag->ready, 1,
                           cudaStreamWriteValueDefault);

    // wait for consumer done before reusing buffer
    cudaStreamWaitValue32(stream, &d_flag->done, 1,
                          cudaStreamWaitValueEq);
}

// ---------- 进程 B: 打开共享的 device memory ----------
void consumer_with_ipc_mem() {
    cudaSetDevice(0);

    // 打开 IPC 内存句柄
    cudaIpcMemHandle_t memHandle;
    // ... (recv memHandle via shm/socket) ...

    SyncFlag* d_remote_flag;
    cudaIpcOpenMemHandle((void**)&d_remote_flag, memHandle,
                         cudaIpcMemLazyEnablePeerAccess);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // wait ready → consume → signal done
    cudaStreamWaitValue32(stream, &d_remote_flag->ready, 1,
                          cudaStreamWaitValueEq);
    consume_kernel<<<1, 256, 0, stream>>>(/*...*/);
    cudaStreamWriteValue32(stream, &d_remote_flag->done, 1,
                           cudaStreamWriteValueDefault);
}
```

### 3.4 三种同步方式总结

|机制|进程内/跨进程|阻塞位置|通信量|典型场景|
|---|---|---|---|---|
|`dist.barrier()` (NCCL)|跨进程|CPU + GPU|allreduce 全量|分布式训练全体同步|
|`cudaStreamWriteValue/WaitValue`|进程内|仅 GPU stream|32-bit 写|同进程多 GPU 流水线|
|`cudaIpcEvent` + `cudaStreamWaitEvent`|跨进程|仅 GPU stream|IPC handle (64B)|跨进程 GPU pipeline|
|IPC Memory + Stream Value|跨进程|仅 GPU stream|32-bit 写|跨进程细粒度信号量|

### 3.5 对称内存（Symmetric Memory）：跨进程共享 Device Memory Pool

前面 3.3 节用 `cudaIpcGetMemHandle` 共享一块 device memory 做信号量，但两个进程看到的是**同一块物理内存的不同虚拟地址**，而且只能共享预先分配好的固定大小 buffer。

CUDA 12.x 的 **Symmetric Memory**（对称内存）更进一步：通过 `cudaMemPool` + IPC，两个进程共享的是**整个内存池**。从各自的池里分配出来的内存，自动映射到同一块物理内存且**虚拟地址偏移相同**（对称）。这意味着不仅可以用 `cudaStreamWriteValue32/WaitValue32` 做同步，两个进程还能在同一块 device memory 上做更复杂的协作。

#### 核心原理

```text
进程 A                              进程 B
───────                             ───────
cudaMemPoolCreate(&poolA)           (接收 poolHandle)
cudaMemPoolExportToShareableHandle  cudaMemPoolImportFromShareableHandle
  → poolHandle ──── shm ─────────→    → poolB (和 poolA 是同一个物理池)

cudaMallocFromPoolAsync(&ptrA,      cudaMallocFromPoolAsync(&ptrB,
    size, poolA, streamA)               size, poolB, streamB)

ptrA 和 ptrB 指向 同一块物理内存，offset 对称
↓
 GPU A: cudaStreamWriteValue32(streamA, ptrA, 1)
 GPU B: cudaStreamWaitValue32(streamB,  ptrB, 1)
```

与 2.2 节用 pinned host memory 做信号量的关键区别：

- Pinned host memory：信号走 PCIe 来回，延迟 ~μs 级
- 对称内存 device memory：信号在 NVLink/NVSwitch 上走，延迟 ~ns 级

#### 完整代码示例

```cpp
// ============ 进程 A: 创建并导出内存池 ============
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

void process_a() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Step 1: 创建内存池
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = 0;
    // 让 pool 可以跨进程共享
    poolProps.flags = cudaMemPoolCreateReleaseThresholdDefault;

    cudaMemPool_t pool;
    cudaMemPoolCreate(&pool, &poolProps);

    // Step 2: 导出 pool 的 IPC handle
    int fd;
    cudaMemPoolExportToShareableHandle(
        &fd, pool, cudaMemHandleTypePosixFileDescriptor, 0);

    // Step 3: 通过 POSIX 共享内存把 fd 发给进程 B
    int shm = shm_open("/pool_fd_shm", O_CREAT | O_RDWR, 0666);
    ftruncate(shm, sizeof(int));
    void* shm_ptr = mmap(NULL, sizeof(int),
                         PROT_READ | PROT_WRITE, MAP_SHARED, shm, 0);
    memcpy(shm_ptr, &fd, sizeof(int));

    // Step 4: 从自己的 pool 里分配信号内存
    uint32_t* sync_flag;
    cudaMallocFromPoolAsync((void**)&sync_flag, sizeof(uint32_t), pool, stream);
    cudaMemsetAsync(sync_flag, 0, sizeof(uint32_t), stream);

    // Step 5: 做计算 → 写信号到对称内存
    produce_kernel<<<1, 256, 0, stream>>>(/* ... */);
    cudaStreamWriteValue32(stream, sync_flag, 1,
                           cudaStreamWriteValueDefault);
    printf("Process A: signaled via symmetric memory\n");

    cudaStreamSynchronize(stream);

    munmap(shm_ptr, sizeof(int));
    close(shm);
    cudaFreeAsync((void*)sync_flag, stream);
    cudaStreamDestroy(stream);
    cudaMemPoolDestroy(pool);
}

// ============ 进程 B: 导入内存池 ============
void process_b() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Step 1: 从共享内存读取 fd
    int shm = shm_open("/pool_fd_shm", O_RDONLY, 0666);
    int fd;
    pread(shm, &fd, sizeof(int), 0);

    // Step 2: 导入 pool —— 现在 poolB 和进程 A 的 poolA 是同一个物理池
    cudaMemPool_t pool;
    cudaMemPoolImportFromShareableHandle(
        &pool, (void*)(uintptr_t)fd, cudaMemHandleTypePosixFileDescriptor, 0);

    // Step 3: 从同一个物理池分配 —— 拿到和进程 A 对称的内存
    uint32_t* sync_flag;
    cudaMallocFromPoolAsync((void**)&sync_flag, sizeof(uint32_t), pool, stream);

    // Step 4: 等进程 A 的 GPU 写信号 → 再做计算
    cudaStreamWaitValue32(stream, sync_flag, 1,
                          cudaStreamWaitValueEq);
    consume_kernel<<<1, 256, 0, stream>>>(/* ... */);
    printf("Process B: woken up by symmetric memory signal\n");

    cudaStreamSynchronize(stream);

    close(shm);
    shm_unlink("/pool_fd_shm");
    cudaFreeAsync((void*)sync_flag, stream);
    cudaStreamDestroy(stream);
    cudaMemPoolDestroy(pool);
}
```

#### 对称内存 vs 普通 IPC Memory

|特性|IPC Memory (`cudaIpcGetMemHandle`)|Symmetric Memory (`cudaMemPool` + IPC)|
|---|---|---|
|共享粒度|单块已分配的 buffer|整个内存池（可动态分配/释放）|
|虚拟地址|两个进程看到**不同** VA|两个进程看到**相同** VA（对称）|
|指针能否直接传递|否，需要各自 `cudaIpcOpenMemHandle`|是，offset 一致，可直接交换指针偏移|
|分配灵活性|只能共享预先分配好的固定大小|双方可以各自 `cudaMallocFromPoolAsync`/`cudaFreeAsync`|
|CUDA 版本要求|CUDA 4.0+|CUDA 12.0+|
|适用场景|固定 buffer 的一次性共享|长期协作、动态分配、复杂数据结构共享|

## 4. 基于nvshmem

nvshmem中提供了device side的api用于发出信号，等待信号，可以嵌入到kernel中进行同步

//TODO