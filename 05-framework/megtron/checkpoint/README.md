# Megtron checkpoint 实现

megtron实现的checkpoint机制叫做torch_dist，支持

- Sharding：分片存储
- Async Save： 异步持久化
- 弹性拓展：并行策略和机器数量灵活变化

## 1. 核心理念

torch_dist是Megatron自研的分布式checkpoint机制，核心目标是支持大规模模型在复杂并行策略下的高效checkpoint管理和弹性扩展。

## 2. sharding

ShardedTensor核心属性：

- key: 张量的唯一标识
- data: 本地数据分片
- dtype, global_shape: 全局张量的元信息
- local_shape, global_offset: 本地分片的位置信息
- replica_id: 副本ID（用于DP维度）
- flattened_range: 扁平化索引范围

## 3. Async Save

```cpp
# 1. 主训练进程准备state dict
state_dict = prepare_sharded_state_dict()

# 2. 创建异步请求（数据拷贝到CPU或pin memory）
async_request = dist_checkpointing.save(
    state_dict,
    checkpoint_dir,
    async_sharded_save=True  # 启用异步
)

# 3. 训练进程立即继续，后台worker执行实际保存
# - 使用独立进程/线程池
# - 通过共享内存或队列传递数据
# - 异步写入存储系统

# 4. 训练进程可选择性等待完成
async_request.wait()  # 可选
```

## 4. Elastic Scaling

