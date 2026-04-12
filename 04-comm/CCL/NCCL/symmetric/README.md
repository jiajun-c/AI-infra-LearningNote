# NCCL 共享内存

NCCL通过注册共享内存实现虚拟地址映射，让GPU可以直接以Load/Store指令访问NVLink内的所有显存。开启后可以显著降低集合通信的时间。

该特性在NCCL2.27被引入，在之前GPU也可以看到对方的物理显存，但是需要通过cuda IPC API来交换内存句柄(Handle)来建立映射

- 在此时的数据流转路径是 User Buffer -> 本地通信buffer -> 远端通信buffer -> 远端user buffer
- SymMem模型，依托于新硬件(Hopper/BlackWell)的MultiMem硬件指令和新的内存机制，将其映射到一个虚拟地址空间中，GPU0的sm只需要发出一句底层的store命令就可以顺着卡到达其他所有卡的User Buffer中

