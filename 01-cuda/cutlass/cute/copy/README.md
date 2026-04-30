# 如何使用cute进行数据拷贝

cute进行数据拷贝能力主要基于两点

- tile：对数据进行划分，确定每个block/thread要负责的数据范围
- copy：对数据进行拷贝


```cpp
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord)   // coord to slice into "remainder"
{
  return inner_partition(static_cast<Tensor&&>(tensor),
                         tiler,
                         coord);
}
```