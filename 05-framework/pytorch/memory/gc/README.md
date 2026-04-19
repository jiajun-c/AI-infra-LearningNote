# Python GC 机制

Python中有两类机制

1. 引用计数
2. 垃圾回收

引用计数的原理是当引用计数归零的时候，对象会立刻释放

但是对于循环引用的情况，python自己无法回收，需要调用gc.collect()

对于闭包的引用，del后也不会谨慎释放，需要使用`callbacks.clear()`，长时间运行脚本内存会积累，需要定期调用`gc.collect()`

`gc.collect` 和 `empty_cache` 的区别在于 gc面向的是对象，`empty_cache`面向的是空闲缓存看