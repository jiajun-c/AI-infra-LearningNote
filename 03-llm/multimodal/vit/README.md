# VIT 算法

vit算法的目的是将transformer对于文本的理解迁移到图像上

vit算法的流程可以分为下面的几部分
- patch embedding：将输入图片划分为固定大小的patch
- positional encoding：加入位置编码，但是和文本的位置编码有区别
- LN/multi-head attention/LN
- MLP