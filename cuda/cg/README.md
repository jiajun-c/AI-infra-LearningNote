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


集群组可以将若干个thread_block组织为一个cluster_block, 将原本 grid->block 的形式转换为 grid -> cluster -> block, `__global__ void __cluster_dims__(2, 1, 1)` 将x轴上的两个block组织为一个cluster。


集群组接口可以获取线程ID和块ID

- thread_rank() 获取cluster中的线程ID
- block_rank() 获取块ID
- thread_count() 获取cluster中的线程数量
- block_count() 获取集群中的块数量

如下所示，使用block_rank获取ID，

```cpp
#include <cstdio>
#define _CG_HAS_CLUSTER_GROUP
#include <cooperative_groups.h>
#include <iostream>

using namespace std;

namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) 
simple_kernel() {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    if (threadIdx.x == 0) {
        printf("block rank %d\n", rank);
    }
}

int main() {

    simple_kernel<<<4, 32>>>();
    cudaDeviceSynchronize();
}
// 输出
// block rank 0
// block rank 1
// block rank 0
// block rank 1
```

来查看剩下两个接口

```cpp
__global__ void __cluster_dims__(2, 1, 1) 
simple_kernel() {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int trank = cluster.thread_rank();
    int thread_count = cluster.num_threads();
    int block_count = cluster.num_blocks();
    unsigned int rank = cluster.block_rank();
    if (threadIdx.x == 0) {
        printf("thread count %d block count %d\n", thread_count, block_count);
        printf("block rank %d %d\n", rank, trank);
    }
}

// thread count 64 block count 2
// thread count 64 block count 2
// thread count 64 block count 2
// thread count 64 block count 2
// block rank 0 0
// block rank 1 32
// block rank 0 0
// block rank 1 32
```

集群组内部需要进行数据的同步，在这里主要解决`mbarrier`指令进行实现，cuda中对其进行了一些封装`cluster.barrier_arrive()` 表示已经执行到了这里，但是不强制进行同步，仍然可以执行下面的指令，直到`barrier_wait`才会强制数据进行同步。相比于syncthreads，这样的开销更小，同时也可以通过在中间插入一些计算，来隐藏延迟。

```cpp
    cluster.barrier_arrive(); 

    // 4. 延迟隐藏 (Latency Hiding)
    // 在等待其他 Block 准备好的间隙，做一些本地计算
    local_processing(block);

    // 5. 【核心知识点】映射远程 Shared Memory
    // 目标：读取下一个 Rank 的数据 (Ring Pattern)
    // Rank 0 -> 读 Rank 1
    // Rank 1 -> 读 Rank 0
    unsigned int neighbor_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
    
    // map_shared_rank: 
    // 替代了之前复杂的 __cvta_generic_to_shared + set_block_rank + cast
    // 它直接返回一个指向 neighbor_rank 的 array[0] 的合法指针
    int *dsmem = cluster.map_shared_rank(&array[0], neighbor_rank);

    // 6. 阻塞等待 (Blocking Wait)
    // 确保所有其他 Block 都执行到了 barrier_arrive()
    // 这意味着它们的 Shared Memory 已经初始化完毕，可以安全读取了
    cluster.barrier_wait();

    // 7. 消费数据
    process_shared_data(block, dsmem, debug_out);

    // 8. 全局同步 (防止某些 Block 跑太快退出了，导致 Shared Memory 被回收)
    cluster.sync();
```

接下来继续介绍分布式共享内存接口，这个是hopper架构之后引入的一个新特性，之前的共享内存访问仅限于当前的线程块内，分布式共享内存当前线程块可以访问其他线程的共享内存。

如下所示，通过map_shared_rank获取到对应线程块所拥有的共享内存

```cpp
    unsigned int neighbor_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
    
    // map_shared_rank: 
    // 替代了之前复杂的 __cvta_generic_to_shared + set_block_rank + cast
    // 它直接返回一个指向 neighbor_rank 的 array[0] 的合法指针
    int *dsmem = cluster.map_shared_rank(&array[0], neighbor_rank);
```

完整的代码见`dsmem.cu` 中