# 基于分布式共享内存的算子优化

我们来一个算子其对一个输入向量的两半进行相加，这一点在TP并行等情况下很常见，比如在TP并行下，一个线程块的输入向量被切分为两半，那么两个线程块的输入向量相加，就可以使用分布式共享内存进行优化。

在不使用分布式共享内存的版本中其需全部从HBM中读取数据，如下所示

```cpp
template <Stage stage>
__global__ void __cluster_dims__(2, 1, 1) cluster_reduce_baseline(half* input, half* output, int size) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    extern __shared__ half smem[];

    int my_rank = cluster.block_rank();
    int neighbor_rank = (my_rank + 1) % cluster.num_blocks();
    int tid = block.thread_rank();
    int num_threads = block.num_threads();

    // 模拟：数据加载到 SMEM
    for (int i = tid; i < size; i += num_threads) {
        smem[i] = input[my_rank * size + i];
    }
    cluster.sync(); 

    // 计算：访问 Global Memory (HBM)
    for (int i = tid; i < size; i += num_threads) {
        half val_local = smem[i];
        
        // <--- 关键点：必须回 HBM 去读邻居的数据
        // 假设邻居已经把数据准备好在 Input 中 (模拟最理想情况)
        // 在真实场景中，邻居可能还需要先 Write HBM，这里我们只计算 Read HBM 的开销
        half val_remote = input[neighbor_rank * size + i]; 

        if constexpr (stage == Stage::LINEAR) {
            smem[i] = __hadd(val_local, val_remote);
        } else if constexpr (stage == Stage::FFN) {
            half sum = __hadd(val_local, val_remote);
            half zero = __float2half(0.0f);
            smem[i] = __hgt(sum, zero) ? sum : zero;
        }
    }
    cluster.sync();

    // Store
    for (int i = tid; i < size; i += num_threads) {
        output[my_rank * size + i] = smem[i];
    }
}
```

如果我们使用分布式共享内存，那么我们可以使用通过访问其他cluster共享内存的方式来进行优化

```cpp
template <Stage stage>
__global__ void __cluster_dims__(2, 1, 1) cluster_reduce_dsmem(half* input, half* output, int size) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    extern __shared__ half smem[];

    int my_rank = cluster.block_rank();
    int tid = block.thread_rank();
    int num_threads = block.num_threads();

    // 模拟：数据已经在 Shared Memory 中 (Load from Global)
    for (int i = tid; i < size; i += num_threads) {
        smem[i] = input[my_rank * size + i];
    }
    cluster.sync(); 

    // DSMEM 映射
    int neighbor_rank = (my_rank + 1) % cluster.num_blocks();
    half* neighbor_smem = cluster.map_shared_rank(&smem[0], neighbor_rank);

    // 计算：访问 DSMEM
    for (int i = tid; i < size; i += num_threads) {
        half val_local = smem[i];
        half val_remote = neighbor_smem[i]; // <--- 关键点：走 Cluster Network

        if constexpr (stage == Stage::LINEAR) {
            smem[i] = __hadd(val_local, val_remote);
        } else if constexpr (stage == Stage::FFN) {
            half sum = __hadd(val_local, val_remote);
            half zero = __float2half(0.0f);
            smem[i] = __hgt(sum, zero) ? sum : zero;
        }
    }
    cluster.sync();

    // Store
    for (int i = tid; i < size; i += num_threads) {
        output[my_rank * size + i] = smem[i];
    }
}
```

测试结果对比

```cpp
Benchmark Configuration:
  Elements per Block: 16384 (FP16)
  SMEM Usage: 32 KB
  Hardware: Requires H100 (SM90)

[Baseline (Global Mem)] Avg Time: 18.5940 us
[DSMEM (Cluster Mem)] Avg Time: 15.0444 us

Verification: PASSED
```