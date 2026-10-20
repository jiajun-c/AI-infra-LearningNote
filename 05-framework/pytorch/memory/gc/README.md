# Python GC 机制

Python的GC机制分为两个部分
  
- CPython的引用计数GC
- Python GC 模块的分代GC

## 1. 引用计数

每个对象通过refcnt来判断当前对象是否存在引用，refcnt=0的时候进行回收

## 2. 分代GC

但是当Python的对象中存在循环引用时，将无法被顺利gc，这种时候需要使用分代gc，他会检查对象的不可达状态，假设循环引用但是已经不可达，那么可以进行gc

分代gc是将python对象分为三类

- 年轻代：新创建，需要GC跟踪的对象，被回收频率最高
- 中年代：从年轻代活下来的对象，回收频率较低
- 老年代：多次存活晋升下来的对象，回收频率最低

分代gc的触发是根据阈值来判断的

```python
import gc

print(gc.get_threshold())
```

输出为700, 10, 10，表示三代分别新增多少后会触发

### 2.1 GC 库的API

手动调用gc，返回结果表示此次发现的不可达对象数量

```python
import gc
n = gc.collect()
print(n)
```

也可以使用gc.collect(n) n = 0,1,2来表示对年轻到老的回收

### 2.2 GC的性能优化

对于高频分配对象的场景，不关闭gc可能会导致影响性能

如`gc_disable.py`中的代码

```cpp
def process_records(count: int, batch_size: int = 2_000) -> int:
    """Build acyclic containers in batches and return a stable checksum."""
    checksum = 0
    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        records = []
        for i in range(batch_start, batch_end):
            records.append({
                "id": i,
                "values": [i, i + 1, i + 2],
                "metadata": {"valid": i % 2, "group": i % 7},
            })

        # Keep this batch alive while it is processed. The tracked containers
        # accumulate long enough to cross GC thresholds, then refcounting frees
        # them when the batch goes out of scope. No references point in a cycle.
        for record in records:
            checksum += sum(record["values"])
            checksum += record["metadata"]["valid"] + record["metadata"]["group"]
    return checksum
```

