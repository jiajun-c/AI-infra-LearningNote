# 异步数据拷贝

## 1. 底层原语

使用cp.async原语进行异步数据拷贝，假设不使用 `cp.async` 时，数据将会从全局内存->L2->L1->寄存器->共享内存，

假设使用 cp.async 时，存在两种策略
- `cp.async.cg` 数据将会从全局内存->L2->共享内存
- `cp.async.ca` 数据将会从全局内存->L2->L1->共享内存，表示在各个层级进行缓存，只是不经过寄存器

## 2. memcpy_async

`memcpy_async` 是 CUDA 提供的异步数据拷贝函数，其底层原理是使用 `cp.async` 进行实现，可以通过 cooperative_groups 进行调用

