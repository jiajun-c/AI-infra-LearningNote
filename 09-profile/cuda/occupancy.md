# GPU Occupancy

## 1. 定义

Occupancy = 一个sm上实际活跃的warp数量/该sm硬件最大支持的warp数量

- 现代GPU每个sm的线程数量是有限的，例如blackwell上面的限制是2048，而Ampere是1536
- 假设kernel中每个sm实际驻留32个warp，那么occupancy为50%

## 2. 作用

GPU上的延迟掩盖是通过warp的调度来掩盖延迟，假设一个warp需要去等待HBM的访存，那么此时可以调度其他的warp

## 3. occupancy限制

- 每个线程的寄存器的数量
- 每个block的共享内存的数量
- SM的warp上限
