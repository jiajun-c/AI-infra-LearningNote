# cuda PTX 内嵌汇编

PTX 是 CUDA 的一种内嵌汇编语言，CUDA 运行时库会编译 PTX 代码，并生成相应的 PTX 汇编代码。

对应的语法如下所示, `asm` 关键词后面放入汇编代码，`%idx` 表示是后面的第几个传入的参数，`"=r"(laneid)` 表示传入 laneid 作为参数

## PTX 修饰符含义

- `r` 表示寄存器
- `f` 浮点寄存器
- `l` 表示一个64位的指针地址
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

使用`::"memory"`来保证操作数在执行这条语句之前所有的访存操作都完成。

```cpp
asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
```

### 使用PTX进行高效计时

如下所示，在开始和结束两个时间段将当前时钟周期存入到变量中，然后计算出操作所消耗的时钟周期。

```cpp
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	for (int j=0 ; j < REPEAT_TIMES ; ++j) {
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
```