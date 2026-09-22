# Data Parallel

## 1. 前向

数据并行的前向过程中保存了对应每个batch的激活值

## 2. 反向

在数据并行的反向过程需要进行通信，对通信的梯度进行allreduce
