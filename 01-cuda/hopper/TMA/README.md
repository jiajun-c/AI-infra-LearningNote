# TMA(Tensor memory access)

TMA是hopper架构下加入的新特性，其数据通路为

Global Memory (DRAM) $\rightarrow$ L2 Cache $\rightarrow$ TMA Hardware Engine $\rightarrow$ Shared Memory (SRAM)

其和cp.async的区别有两点
- 走的是DMA而不是通过LD/ST数据通路，使得其不会去阻塞其他的访存请求
- 只需要由一个CTA中一个线程去发起请求就可以获取到整个CTA所需要的数据量

TMA要求起始的地址是16字节(128b)对齐，对应的多维度tensor的stride需要保证是16字节对齐的

