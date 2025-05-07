# Einops

einops可以以一种简单易读的方式来实现易读并可靠的代码

## 1. 转换张量维度

如下所示输入是(3, 4, 5)，将最后两个维度进行交换

```python3
output = einops.rearrange(input_tesnor, 'i j k -> i k j')
print(output.shape)   
```

也可以针对其中的某个维度进行拆分

如下所示，
```python3
output = einops.rearrange(input_tesnor, 'i (a b) k -> i a b k', a = 2, b = 2)
```

## 2. 求和

如下所示使用`einsum`进行求和，如下所示是进行一个矩阵乘法的操作

```python3
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = einops.einsum(a, b,'i j, j k -> i k')
print(c.shape)
```

