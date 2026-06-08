# 内存层级

GPU内存层级从块到慢，从小到大

- 寄存器
  - 每个线程私有，速度最快(~1 cycle)
  - 容量最小
  - 溢出到local memory会严重影响性能
- Shared Memory/L1 Cache
  - 同一sm内的线程块共享
  - 延迟 ~20-30cycle
  - 可编程管理(shared memory) + 硬件管理(L1 cache)，两者共享同一物理SRAM
  - 典型大小：每个SM 128~228 KB

- L2 cache
  - 所以SM共享
  - 延迟~200cycles
  - 典型大小：几MB到几十MB
- Global Memory(HBM)
  - 所有线程可访问，容量最大
  - 延迟~400-600cycles，带宽~2-3TB/s
  - 典型几十GB