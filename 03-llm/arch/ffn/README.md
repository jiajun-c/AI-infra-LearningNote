# FFN

## 1. 定义

$FFN(x) = W_2 \cdot (ReLU(W_1 \cdot x))$

- 升维：乘$W_1$，$d\to 4d$ (扩大四倍)
- 激活：ReLU，过滤掉负值
- 将维：乘$W_2$，$4d\to d$ (缩回原维度)

## 2. GeGLU

