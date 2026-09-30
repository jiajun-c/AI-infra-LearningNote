# 分布式共享内存

分布式共享内存可以直接在这个CTA上去访问其他CTA上的共享内存，通过`map_dsmem_ptr`接口来获取到cluster中其他CTA上的共享内存

```python
if lane == 0:
    for peer in range(cluster_size):
        remote_tile = cute.arch.map_dsmem_ptr(local_tile, cutlass.Int32(peer))
        value = remote_tile[0]
        output[rank * cluster_size + peer] = value
```