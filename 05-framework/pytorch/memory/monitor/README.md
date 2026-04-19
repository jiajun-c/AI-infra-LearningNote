# 显存监控

## 1. 相关接口

如下所示，allocated 表示实际分配的内存，reserved表示在缓存块里的

```python
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
peak = torch.cuda.memory.max_memory_allocated() / 1024**3
```

一个简单的helper函数

```python
def snapshot_memory(prefix=""):
    """打印当前显存使用情况"""
    gc.collect()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"[{prefix}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    return allocated, reserved
```