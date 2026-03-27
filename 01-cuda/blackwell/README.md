# BlackWell架构

## 1. UMMA

在Blackwell架构上引入了UMMA，其替代了前几代的MMA指令，最大的特点为操作数可以来自不同的存储层级

- SS：A和B都来自共享内存
- TS：A来自于TMem而B来自于共享内存

Tmem的大小有256KB，可以用于将B持久化存储

## 2. 双SM协同处理数据

支持两个SM协同处理一条UMMA指令

- LeaderCTA：实际执行MMA指令
- PeerCTA：配合提供数据
- B 矩阵在两个 SM 间分区：实际 SMEM 中 B 的维度 = N / ClusterM = 64 / 2 = 32
- 这使得单次 MMA 操作可处理更大的 tile 尺寸（256×64×32）

## 3. TMA的变化

TMA Load的终点除了共享内存，还增加到了TMEM

