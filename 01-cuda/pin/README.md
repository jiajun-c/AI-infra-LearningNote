# Pin memory 

cuda pin memory可以分配一片锁页内存，使用
`cudaMallocHost`或者`cudaHostRegister`进行分配

如果是`cudaMallocHost`，可以cuda直接分配，保证自动对齐，

`cudaMallocHost` 性能测试，可以看到时间随着大小线性增长，每4K页面的就会触发一次page fault，这是一个很昂贵的操作

```shell
1 MB        alloc:    0.74 ms  free:    0.22 ms
16 MB       alloc:    4.76 ms  free:    1.27 ms
64 MB       alloc:   18.80 ms  free:    5.97 ms
128 MB      alloc:   36.93 ms  free:   12.63 ms
256 MB      alloc:   74.50 ms  free:   24.92 ms
1024 MB     alloc:  295.11 ms  free:   84.26 ms
2048 MB     alloc:  576.93 ms  free:  157.79 ms
4096 MB     alloc: 1147.77 ms  free:  316.68 ms
```

`cudaMalloc` 性能测试，可以发现随着大小的增加时间没有明显增加，但是free的时间增加较多。

```shell
size        alloc             free            
----        -----             ----            
1 MB        alloc:    0.13 ms  free:    0.10 ms
16 MB       alloc:    0.08 ms  free:    0.08 ms
64 MB       alloc:    0.08 ms  free:    0.09 ms
256 MB      alloc:    0.16 ms  free:    0.13 ms
1024 MB     alloc:    0.63 ms  free:    0.36 ms
2048 MB     alloc:    1.06 ms  free:    0.53 ms
4096 MB     alloc:    0.56 ms  free:    0.95 ms
```

pre-fault 版本去优化pin memory申请，使用多线程去发起页中断去建立虚拟页表和物理页表之间的联系，这部分多线程下性能更佳，但是register部分无法加速

这部分的流程是
- 查询每个物理页的地址（调用 get_user_pages）
- 将页地址注册到 CUDA 驱动的 pin table
- 整个过程持有一把全局锁（CUDA 驱动内部的 pin_lock）
- pre-fault 线程数越多，物理页分布越碎片化（每个线程分配的页在不同 NUMA 节点或内存区域），导致 cudaHostRegister 遍历页表的开销增大。



```shell
size        threads     fault                   register            total           
--------------------------------------------------------------------------------
[64 MB]
  64 MB       threads:  1  fault:   20.78 ms  register:   4.67 ms  total:   25.45 ms
  64 MB       threads:  2  fault:   10.84 ms  register:   3.31 ms  total:   14.16 ms
  64 MB       threads:  4  fault:    5.68 ms  register:   3.44 ms  total:    9.13 ms
  64 MB       threads:  8  fault:    3.35 ms  register:   3.77 ms  total:    7.12 ms
  64 MB       threads: 16  fault:    3.54 ms  register:   4.09 ms  total:    7.63 ms
  64 MB       threads: 32  fault:    5.66 ms  register:   5.05 ms  total:   10.72 ms
  64 MB       threads: 64  fault:    5.90 ms  register:   4.41 ms  total:   10.32 ms

[256 MB]
  256 MB      threads:  1  fault:   83.32 ms  register:  14.33 ms  total:   97.64 ms
  256 MB      threads:  2  fault:   41.00 ms  register:  11.80 ms  total:   52.81 ms
  256 MB      threads:  4  fault:   21.59 ms  register:  11.93 ms  total:   33.51 ms
  256 MB      threads:  8  fault:   13.18 ms  register:  13.01 ms  total:   26.19 ms
  256 MB      threads: 16  fault:   12.19 ms  register:  13.42 ms  total:   25.61 ms
  256 MB      threads: 32  fault:   15.84 ms  register:  17.45 ms  total:   33.29 ms
  256 MB      threads: 64  fault:   20.76 ms  register:  19.50 ms  total:   40.25 ms

[1024 MB]
  1024 MB     threads:  1  fault:  304.10 ms  register:  66.74 ms  total:  370.83 ms
  1024 MB     threads:  2  fault:  164.87 ms  register:  43.92 ms  total:  208.79 ms
  1024 MB     threads:  4  fault:   86.66 ms  register:  39.53 ms  total:  126.19 ms
  1024 MB     threads:  8  fault:   51.07 ms  register:  48.59 ms  total:   99.66 ms
  1024 MB     threads: 16  fault:   44.62 ms  register:  51.01 ms  total:   95.63 ms
  1024 MB     threads: 32  fault:   56.71 ms  register:  67.62 ms  total:  124.32 ms
  1024 MB     threads: 64  fault:   61.08 ms  register:  78.58 ms  total:  139.66 ms

[4096 MB]
  4096 MB     threads:  1  fault: 1204.95 ms  register: 275.37 ms  total: 1480.32 ms
  4096 MB     threads:  2  fault:  661.11 ms  register: 237.05 ms  total:  898.16 ms
  4096 MB     threads:  4  fault:  342.09 ms  register: 258.97 ms  total:  601.06 ms
  4096 MB     threads:  8  fault:  205.57 ms  register: 198.00 ms  total:  403.57 ms
  4096 MB     threads: 16  fault:  170.82 ms  register: 253.02 ms  total:  423.83 ms
  4096 MB     threads: 32  fault:  219.80 ms  register: 275.30 ms  total:  495.09 ms
  4096 MB     threads: 64  fault:  223.34 ms  register: 280.66 ms  total:  503.99 ms
(base) ➜  pin git:(main) ✗ 
```