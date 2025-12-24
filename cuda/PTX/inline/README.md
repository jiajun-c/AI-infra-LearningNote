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

## 使用PTX进行高效计时

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

## 数据移动和转换指令

### 1. 数据缓存操作

用于数据加载的缓存指令

`.ca`: cache all levels，使用正常的缓存策略在L1, L2层级进行数据的分配，保证了在L2层级的数据局部性，但是在多个L1层级上不能保证和全局数据的一致性
`.cg`: cache global level，在L2及以下的层级进行数据缓存
`.cs`: cache streaming：适用于只访问一两次的数据，会被最先替换出cache
`.lu`: last use,避免不必要的写回到不会再使用缓存行上
`.cv`: 不在再进行缓存或者读取，会把L2中对应的缓存行给抛弃

用于数据存储的缓存操作

`.wb`: write back, 将数据写入到缓存的所有层级中
`.cg`: cache global level，在L2及以下的层级进行数据缓存
`.cs`: cache streaming：虽然被写入到了cache中，但是会被最先替换出cache
`.wt`: write through, 直接写入到了全局地址空间中，不通过L2


### 2. 缓存替换策略

`evict_normal`: 默认的缓存替换策略
`evict_first`: 通过该策略进行缓存的将会被首先驱逐出cache
`evict_last`: 通过该策略进行缓存的将会被最后驱逐出cache
`evict_unchange` 不改变数据驱逐的优化级
`no_allocate`: 不为数据分配缓存行

