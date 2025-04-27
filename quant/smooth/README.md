# 激活值平滑

在transformer中，权重的值往往是稳定的，而激活值是不稳定的。

其核心公式如下所示

$Y = X\cdot W = (X\cdot diag(s)^{-1})\cdot (diag(s)\cdot W)$

其中Y为输出，X为输入的激活值，W为权重值。

$s_j = \max(|W_{:,j}|)$

相当于对X和W同时进行平滑

