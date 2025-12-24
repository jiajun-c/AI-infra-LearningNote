# 地址空间判定函数

## 1. 判断空间类型

判断是否为全局地址空间和共享内存
```cpp
__device__ unsigned int __isGlobal(const void *ptr);
__device__ unsigned int __isShared(const void *ptr);
```


