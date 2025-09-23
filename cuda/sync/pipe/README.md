# CUDA 流水线接口

`cuda::pipeline` 接口可以实现协调机制，将异步操作如 `cuda::memcpy_async`按阶段进行排序，通过该方法可以实现双缓冲。

线程使用下面的四个接口和pipeline对象进行交互
- 获取当前所在的阶段
- 提交一些操作到当前的阶段
- 等待之前的操作完成
- 释放pipeline所在的阶段

创建pipe
```cpp
 auto pipeline = cuda::make_pipeline(group, &shared_state);
```

阻塞当前管道直到下个阶段可用
```cpp
pipeline.producer_acquire();
```

将当前线程在此语句之前的工作提交到当前阶段

```cpp
pipeline.producer_commit();
```

消费者等待可用
```cpp
pipeline.consumer_wait();
compute(shared[(subset - 1) % 2]);
pipeline.consumer_release();
```


一个简单的二阶段pipeline例子如下所示

```cpp
template <typename T>
__global__ void example_kernel(T* global0, T* global1, cuda::std::size_t subset_count) {
    extern __shared__ T s[];
    auto group = cooperative_groups::this_thread_block();
    T* shared[2] = { s, s + 2 * group.size() };

    // 创建流水线
    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(group, &shared_state);

    // 初始化流水线
    pipeline.producer_acquire();
    cuda::memcpy_async(group, shared[0],
                       &global0[0], sizeof(T) * group.size(), pipeline);
    cuda::memcpy_async(group, shared[0] + group.size(),
                       &global1[0], sizeof(T) * group.size(), pipeline);
    pipeline.producer_commit();

    // 流水线处理循环
    for (cuda::std::size_t subset = 1; subset < subset_count; ++subset) {
        pipeline.producer_acquire();
        cuda::memcpy_async(group, shared[subset % 2],
                           &global0[subset * group.size()],
                           sizeof(T) * group.size(), pipeline);
        cuda::memcpy_async(group, shared[subset % 2] + group.size(),
                           &global1[subset * group.size()],
                           sizeof(T) * group.size(), pipeline);
        pipeline.producer_commit();
        pipeline.consumer_wait();
        compute(shared[(subset - 1) % 2]);
        pipeline.consumer_release();
    }

    // 处理最后一批数据
    pipeline.consumer_wait();
    compute(shared[(subset_count - 1) % 2]);
    pipeline.consumer_release();
}
```