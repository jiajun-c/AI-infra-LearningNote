# Dtensor

## 1. 用途

dtensor是pytorch的下一代shardTensor，目标是在SPMD的编程模型下，把切分+复制+部分规约，统一成一种张量子类

## 2. 抽象基元

### 2.1 deviceMesh

把一组进程组织成N维逻辑网格，例如(2, 4)表示DPx2 TPx4，每个mesh dim持有一个子proccessGroup

```cpp
mesh = init_device_mesh("cuda", (2, 4))   # 8 个 device
mesh.get_coordinate()                     # 当前 rank 在网格里的坐标
mesh["dp"], mesh["tp"]                    # 命名子网格
```

### 2.2 `Placement` 表示张量在某一 mesh 维上的分布方式

基类Placement是抽象类

- Shard(dim): 把张量dim在该mesh dim上均匀切分
- Replicate()：在该 mesh dim 上每个 rank 都持有一份完整副本
- Partial(reduce_op="sum"): 每个 rank 持有一段"局部结果"，还需要跨 rank reduce_op 才能得到真值

