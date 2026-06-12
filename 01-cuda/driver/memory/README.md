# CUDA Driver API Stream Memory Operations

这三个 API 解决了同一个问题：**在 GPU 侧完成 stream 间的同步，不需要 CPU 介入。**

```text
传统方式 (CPU 介入):                  新方式 (GPU 侧完成):
  stream A 完成                        stream A 完成
      │                                    │
      ▼                                    ▼
  CPU 等 cudaStreamSynchronize         cuStreamWriteValue32 → signal = 1
      │                                    │
      ▼                                    ▼
  CPU 启动 stream B                   stream B: cuStreamWaitValue32(signal == 1)
```

## 1. cuStreamWaitValue32

**在 stream 中插入一个等待点，阻塞后续操作，直到指定内存地址的值满足条件。**

```c
CUresult cuStreamWaitValue32(
    CUstream stream,      // 哪个 stream 上等待
    CUdeviceptr addr,     // 监控的内存地址 (必须是 device memory 的 4 字节对齐地址)
    cuuint32_t value,     // 期望值
    unsigned int flags    // CU_STREAM_WAIT_VALUE_EQ  (= value)
);                        // CU_STREAM_WAIT_VALUE_GEQ (>= value)
                          // CU_STREAM_WAIT_VALUE_AND (addr & value != 0)
                          // CU_STREAM_WAIT_VALUE_NOR (等待 addr 被其他写入覆盖)
                          // CU_STREAM_WAIT_VALUE_FLUSH (配合 remote NVLink 写入)
```

**工作方式**：

```text
stream 中的操作序列:
  [kernel A] → [kernel B] → [cuStreamWaitValue32] → [kernel C]
                                 │
                            GPU 硬件轮询 addr 的值
                            不等于 value → 阻塞 kernel C
                            等于 value → 放行 kernel C
```

**完整示例**：

```cpp
#include <cuda.h>
#include <cstdio>

#define check(e, msg) if (e != CUDA_SUCCESS) { \
    fprintf(stderr, "%s: %d\n", msg, e); exit(1); }

// 场景: streamA 写完数据后通知 streamB
int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    // 分配 device 内存作为 signal
    CUdeviceptr signal;
    cuMemAlloc(&signal, 4);  // 4 bytes = 32 bits
    int zero = 0;
    cuMemcpyHtoD(signal, &zero, 4);  // 初始化为 0

    CUstream streamA, streamB;
    cuStreamCreate(&streamA, 0);
    cuStreamCreate(&streamB, 0);

    // streamA: 做完事后写 signal = 1
    //   (实际中这里是 kernel launch)
    cuStreamWriteValue32(streamA, signal, 1, CU_STREAM_WRITE_VALUE_DEFAULT);

    // streamB: 等 signal == 1 再开始
    cuStreamWaitValue32(streamB, signal, 1, CU_STREAM_WAIT_VALUE_EQ);

    // streamB 的后续操作只有在 signal == 1 时才执行
    //   (实际中这里是依赖 streamA 结果的 kernel)

    cuStreamSynchronize(streamA);
    cuStreamSynchronize(streamB);

    cuMemFree(signal);
    cuCtxDestroy(ctx);
}
```

## 2. cuStreamWriteValue32

**在 stream 中插入一个写入操作：当前面所有操作完成后，向目标地址写入指定值。**

```c
CUresult cuStreamWriteValue32(
    CUstream stream,      // 哪个 stream 上写
    CUdeviceptr addr,     // 目标内存地址 (4 字节对齐)
    cuuint32_t value,     // 要写的值
    unsigned int flags    // CU_STREAM_WRITE_VALUE_DEFAULT
);                        // CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER (不保证对其他 SM 可见)
```

**关键时序**：

```text
streamA:
  [kernel 数据搬完] → [cuStreamWriteValue32(signal=1)]
                           │
                     kernel 完成后, GPU 硬件自动把 1 写到 signal
                     不需要 CPU 参与, 不需要额外 kernel
```

**典型使用：生产者-消费者同步**：

```cpp
// 生产者 stream: 做完事 → 写 flag
// 消费者 stream: 等到 flag → 开始消费
void producer_consumer(CUstream prod, CUstream cons, CUdeviceptr flag) {
    int zero = 0;
    cuMemcpyHtoD(flag, &zero, 4);  // flag = 0

    // 生产者: 发射 kernel 后写 flag = 1
    // myKernel<<<grid, block, 0, prod>>>(...);
    cuStreamWriteValue32(prod, flag, 1, CU_STREAM_WRITE_VALUE_DEFAULT);

    // 消费者: 等 flag == 1
    cuStreamWaitValue32(cons, flag, 1, CU_STREAM_WAIT_VALUE_EQ);
    // consumeKernel<<<grid, block, 0, cons>>>(...);
}
```

## 3. cuStreamBatchMemOp

**把多个 Wait/Write/Flush 操作打包成一次调用，减少 API 开销。**

当你有大量依赖关系时（比如推理引擎中数百个 stream 之间的依赖），逐个调 `cuStreamWaitValue32` 开销很大。`cuStreamBatchMemOp` 把所有操作打包，**一次 API 调用处理所有同步点**。

```c
CUresult cuStreamBatchMemOp(
    CUstream stream,          // 目标 stream
    unsigned int count,       // 操作数量
    CUstreamBatchMemOpParams *paramArray,  // 操作数组
    unsigned int flags        // 0
);
```

**三种操作类型**：

```cpp
// 操作类型 1: WAIT_VALUE_32 (等同于 cuStreamWaitValue32)
// 操作类型 2: WRITE_VALUE_32 (等同于 cuStreamWriteValue32)
// 操作类型 3: FLUSH_REMOTE_WRITES (等待远程 GPU 的写入对本地可见)

typedef union CUstreamBatchMemOpParams_union {
    CUstreamMemOpWaitValueParams   waitValue;   // 等待操作
    CUstreamMemOpWriteValueParams  writeValue;  // 写值操作
    CUstreamMemOpFlushRemoteWritesParams flush; // 刷新远程写入
} CUstreamBatchMemOpParams;

// WaitValue:
typedef struct {
    CUstreamBatchMemOpType operation;  // CU_STREAM_MEM_OP_WAIT_VALUE_32
    CUdeviceptr            address;    // 监控地址
    union {
        cuuint32_t value;              // 期望值
        cuuint64_t pad;
    } alias;
    cuuint32_t flags;                  // CU_STREAM_WAIT_VALUE_EQ / GEQ / AND / NOR / FLUSH
    CUdeviceptr mask;                  // AND 模式下的掩码
} CUstreamMemOpWaitValueParams;

// WriteValue:
typedef struct {
    CUstreamBatchMemOpType operation;  // CU_STREAM_MEM_OP_WRITE_VALUE_32
    CUdeviceptr            address;    // 目标地址
    union {
        cuuint32_t value;              // 要写的值
        cuuint64_t pad;
    } alias;
    cuuint32_t flags;                  // CU_STREAM_WRITE_VALUE_DEFAULT
} CUstreamMemOpWriteValueParams;
```

**完整示例：多路同步**：

```cpp
// 场景: 3 个生产者完成后, 通知 1 个消费者
// prod[0]→flag[0]=1, prod[1]→flag[1]=1, prod[2]→flag[2]=1
// consumer 等 flag[0]==1 && flag[1]==1 && flag[2]==1

void batch_sync_example() {
    CUstream prod[3], cons;
    CUdeviceptr flags[3];
    for (int i = 0; i < 3; i++) {
        cuStreamCreate(&prod[i], 0);
        cuMemAlloc(&flags[i], 4);
        int zero = 0;
        cuMemcpyHtoD(flags[i], &zero, 4);
    }
    cuStreamCreate(&cons, 0);

    // --- 3 个生产者各自写 flag ---
    for (int i = 0; i < 3; i++) {
        // myKernel<<<..., prod[i]>>>(...);
        cuStreamWriteValue32(prod[i], flags[i], 1,
                             CU_STREAM_WRITE_VALUE_DEFAULT);
    }

    // --- 消费者: 批量等待 3 个 flag ---
    CUstreamBatchMemOpParams params[3];
    for (int i = 0; i < 3; i++) {
        params[i].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        params[i].waitValue.address   = flags[i];
        params[i].waitValue.alias.value = 1;
        params[i].waitValue.flags     = CU_STREAM_WAIT_VALUE_EQ;
        // params[i].waitValue.mask    = 0;  // 不用 AND
    }

    // 一次调用, 3 个等待全部插入 stream
    cuStreamBatchMemOp(cons, 3, params, 0);
    // consumeKernel<<<..., cons>>>(...);
}
```

## 4. 为什么不用 cudaEvent？

| | cudaEvent | cuStreamWriteValue/WaitValue |
| --- | --- | --- |
| **同步方式** | CPU 创建 event, GPU 记录, CPU 查询 | **纯 GPU 侧**，通过 device memory 通信 |
| **CPU 开销** | `cudaEventRecord` + `cudaStreamWaitEvent` 有 API 开销 | 0（GPU 硬件自动轮询/写入） |
| **跨 GPU** | cudaEvent 不支持跨设备 | FLUSH_REMOTE_WRITES 支持 NVLink 跨 GPU |
| **灵活性** | 固定模式: record→wait | 自定义值, 多种条件(EQ/GEQ/AND/NOR) |
| **批量** | 逐个 API 调用 | `cuStreamBatchMemOp` 一次批量 |

## 5. 典型应用场景

```text
1. 推理引擎 (vLLM, TensorRT):
   数百个 request 并发, 每个有独立 stream
   → 用 device memory 做 flag, GPU 侧轮询
   → CPU 只负责提交, 不管同步

2. 跨 GPU 同步 (NCCL):
   GPU 0 写数据到 GPU 1 显存
   → cuStreamWriteValue32(NOR) 写 flag
   → GPU 1: cuStreamWaitValue32(FLUSH) 等 flag + 刷新 NVLink

3. Persistent Kernel 模式:
   一个长驻 kernel 在 GPU 上自旋等 flag
   → CPU 通过 cuStreamWriteValue32 给它发信号
   → 不需要 kernel relaunch
```
