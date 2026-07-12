# inkernel trace

## 1. 使用clock进行计时

CUDA 提供了 clock() 和 clock64() 两个内置函数，可以在 kernel 内部直接读取 SM（流多处理器）的时钟周期计数器。通过在执行前后分别读取，就能测量代码段的执行周期数。

```cpp
// 基本用法
__device__ void measure_example() {
    long long int start = clock64();  // 记录起始周期
    // ... 要测量的代码 ...
    long long int end   = clock64();  // 记录结束周期
    long long int cycles = end - start;
    // 转换为时间：cycles / SM频率
}
```

同时为了避免重排指令，需要使用 `__threadfence` 

clock64() 本身是一条很轻量的指令（S2R 读特殊寄存器），开销极小（1~2 个 cycle），不会显著影响测量精度。

## 2. nv iket

是nv推的一个inkernel profile工具，被用在quack里面

