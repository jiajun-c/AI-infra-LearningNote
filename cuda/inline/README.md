# cuda PTX 内嵌汇编

PTX 是 CUDA 的一种内嵌汇编语言，CUDA 运行时库会编译 PTX 代码，并生成相应的 PTX 汇编代码。

对应的语法如下所示, `asm` 关键词后面放入汇编代码，`%idx` 表示是后面的第几个传入的参数，`"=r"(laneid)` 表示传入 laneid 作为参数

## PTX 修饰符含义

- `r` 表示寄存器
- `+` 表示参数可读可写
- `=` 表示操作数是只写型
- 无表示操作数是只读型，仅作为参数进行提供

```cpp
__global__ void sequence_gpu(int *d_ptr, int N) {
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemID < N) {
        unsigned int laneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        d_ptr[elemID] = laneid;
    }
}
```