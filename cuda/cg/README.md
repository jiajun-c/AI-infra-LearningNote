# cuda 协作组

协作组(cooperative_groups)是CUDA9中引入的一个线程协作机制，提供了一种更加灵活，更加结构化的方式来管理和同步线程。其支持warp级别，block级别，多个block同步，自定义子组的形式来进行线程的同步管理。

协作组由下面的几种元素组成
- 用于表示的协作线程组的数据结构
- 获取到cuda launch api隐式设置的线程，线程块等信息的操作
- 将当前线程组划分为新线程组的操作
- 用于数据拷贝和运算的集合操作
- 线程组内线程同步操作
- 获取线程组相关属性
- 用于底层，分组特定的硬件加速操作

## 1. 分组类型

### 1.1 线程块组

```cpp
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
namespace cg = cooperative_groups;

__global__ void kernel(int *input) {
    __shared__ int x;
    cg::thread_block tb = cg::this_thread_block();
    printf("thread_rank %d\n", tb.thread_rank());
    if (tb.thread_rank() == 0) x = (*input);
    tb.sync();
}

int main() {
    int *d_input;
    cudaMalloc(&d_input, sizeof(int));
    kernel<<<2, 32>>>(d_input);
    cudaDeviceSynchronize();
}
```

对于cg 而言,需要保证所有线程都达到同步，才能进行下一步操作, 不能在cg中有元素提前退出

### 1.2 集群组

集群组接口可以获取线程ID和块ID

- thread_rank() 获取线程ID
- block_rank() 获取块ID

