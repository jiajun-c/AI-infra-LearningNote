# 数据集加载优化

## 1. python dataset

`DataLoader`有如下的接口

- batch_size: 每个mini-batch包含样本数量

```cpp
DataLoader(                                                             
    benchmark_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=False,
)
```

里面几个参数都会对性能有所影响

### 1.1 batch_size

每次batch都会有一个固定开销

- Python 函数调用 `__getitem__` 或者 `__getitems__` 的开销
- pickle/IPC（worker进程->主进程)
- collate: 把N个tensor拼成一个tensor
- Index queue同步(主进程->worker)
- H2D 启动一次

假设batch开大可以平摊这个固定开销，同时可以更好地利用H2D的带宽，但是batch太大也会对显存提出挑战

### 1.2 num_worker

num_workder决定的是Pytroch的dataloader把数据加载从主进程分离出去的并行度，然后采用round robin的方式去派发任务，所以适当地提高num_workder的大小可以增加数据的吞吐。但是当 num_workder大于核心的数量或者IPC同步开销之类更大的时候也会导致性能的下降

### 1.3 pin_memory

使用的pin_memory 去优化数据的的H2D的加载

### 1.4 prefetch_factor

让每个worker保持多个in-flight的task

### 1.5 GPU 计算重叠的 H2D

