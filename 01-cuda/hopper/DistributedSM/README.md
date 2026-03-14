# Hopper 架构分布式共享内存

在以往的架构中，shared memory 只存在于Block内部，Block之间无法互相访问shared memory，在Hopper架构中，在L1和L2 cache之间增加了一层 SM-SM 网络，使得Thread Block Cluster内部的SM可以访问其他Block内的共享内存

如下所示，分为几个部分
- 初始化
- 同步
- 获取分布式共享内存
- 处理共享内存
- 同步

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void cluster_kernel() {
    auto cluster = cg::this_cluster();
    extern __shared__ int shm[];
    // 初始化 ... 
    init_local_shared_data(shm);
    // 保证初始化完成
    cluster.sync();
    // ...
    // 获取 Distributed Shared Memory 地址
    int *dsmem = cluster.map_shared_rank(&shm[0], some_rank);
    // 处理 Distributed Shared Memory
    process_shared_data(dsmem);
    // 保证 Shared Memory 可用，避免其他 Thread Block 访问
    // 当前 Thread Block 的 Shared Memory 失败
    cluster.sync();
}
```