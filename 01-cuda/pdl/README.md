# PDL

PDL(programmatic dependent launch) 是一种优化延迟的方式，可以一定程序上去overlap两个有数据依赖的kernel

我们可以将一个kernel内部大致分为

- launch overhead：硬件启动
- prolog: 不依赖于前面的一个kernel
- mainloop：真正的计算
- memory barrier：写回

## 编程范式

假设有两个kernel，kernel1和kernel2

```cpp
__global__ void kernel1() {
    需要在kernel2之前完成的任务

    // 触发下一个kernel
    cudaTriggerProgrammaticLaunchCompletion();
}
```

```cpp
__global__ void kernel2() {
    // Independent work

    // 等待kernel2的计算结束
    cudaGridDependencySyncronize();

    // depent work
}
```
