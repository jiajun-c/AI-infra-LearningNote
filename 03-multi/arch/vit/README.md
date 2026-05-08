# VIT 算法

vit算法的目的是将transformer对于文本的理解迁移到图像上

## 1. VIT算法流程

vit算法的流程
- 将图片切分成多个16x16的patch，再把每个patch投影为固定长度的向量送入Transformer

![alt text](image.png)