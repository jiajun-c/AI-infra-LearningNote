# FSDP

## 1. 原理

FSDP(full share data parallelism)，其不是一次性加载全部的模型，而是按照模型的层或者模块逐个进行计算。在计算完成适合释放，和dp的区别都在于这里

对于多层的结构而言，其不需要把每一层的权重都保存下来，只需要在用到的层进行allGather即可，相比于DP在多层结构下其显存占用更低

## 2. 实现

```python
class FSDPLinear(nn.Module):
    """
    每卡只存 w_shard = W[rank*chunk:(rank+1)*chunk, :]
    forward 时 all-gather 重建完整 W，计算后立即释放
    """
    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.in_features = in_features
        self.out_features = out_features

        # 初始化完整权重，然后只保留本卡分片
        full_w = torch.randn(out_features, in_features) * 0.01
        self.w_shard = nn.Parameter(shard_param(full_w, world_size, rank))
        # bias 不分片，每卡保存完整（也可以分片，此处简化）
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. all-gather：临时重建完整权重
        w_full = all_gather_param(self.w_shard, self.world_size)  # [out, in]

        # 2. 计算
        out = x @ w_full.T + self.bias  # [B, S, out]

        # 3. w_full 是局部变量，函数返回后自动释放（不存入 self）
        return out

```