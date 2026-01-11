# 归一化

做归一化是因为CNN中包含很多隐含层，每层参数会随着训练而改变优化，所以隐层的输入分布总会发生变化，使得其不满足特定的分布。

## 1. Batch Normalization

对一个batch中所有的样本的同一个特征维度进行归一化。假设输入为(B, H) B = batch size, H = hidden size

对于每个特征维度h，计算该维度在整个batch上的均值和方差，再进行归一化，结合可学习参数放缩和平移

当batch数量较少的情况，统计量较小，效果差，而batch过大时候可能超出内存容量

## 2. Layer Normalization

LayerNorm 是大模型中场景的归一化操作，作用是对特征张量按照某一维度或者某几个维度进行0均值，1方差的归一化，后者是缩放和平移变量

![alt text](image.png)

