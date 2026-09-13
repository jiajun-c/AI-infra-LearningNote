# GPU 内存模型和编程

## 1. 内存模型

对于每个GPU而言，其L1是私有的，但是在L2上存在缓存的一致性，假设多个SM的L1同时写出到L2的话，他们会经过crossbar + slice的请求队列/仲裁，形成一个单一串行化点，最终形成一个内存的操作顺序，只是这个顺序我们是不可预测的

对于GPU的内存模型其实我们也存在两个维度，一个是内存序，一个是作用域范围

### 1.1 内存序

- relaxed: 只保证原子性+单变量
- acquire: 读侧栅栏(消费)
- release: 写侧栅栏(发布)
- acq_rel: 一条RMW同时消费+发布
- sc: 顺序一致

### 1.2 作用域

- cta: 一个thread block内
- cluster: 一个thread cluster内(sm90+)
- gpu: 单张GPU内所有线程
- sys：整个系统，所有GPU+host

## 2. 原子操作

原子操作的指令组成如下所示

- atom.<作用域>.<内存序>.<操作>.<类型>

常见的操作有

- cas：compare and swap
- add：累加
- max/min: 最大最小
- and/or/xor 按位更新
- inc/dec: 带边界回绕的增减

