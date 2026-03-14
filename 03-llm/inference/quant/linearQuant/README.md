# 量化

## 1. 线性映射

线性映射中可以被分为对称映射和非对称映射

如下所示是针对fp32 -> int8的对称映射的代码，其通过取绝对值的最大值进行放缩

- 缩放系数S = |x_max|/127
- x_int8 = up(X/S)

进行反量化的时候直接进行 x_int8/scale 即可

```python3
def quantize(x_fp32):
    scale = torch.max(torch.abs(x_fp32))/127.0
    scale += 1e-7
    
    x_int8 = x_fp32/scale
    # 对于正数和负数的情况进行
    x_int8 += 0.5*torch.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(torch.int8)
    return x_int8, scale

def dequantize(x_int8, scale):
    x_fp32 = x_int8.to(torch.float32) * scale
    return x_fp32
```

输出
```shell
input data:  tensor([[1.2000, 2.3000, 3.4000],
        [4.5000, 5.6000, 6.7000]])
quantized data :  tensor([[ 23,  44,  64],
        [ 85, 106, 127]], dtype=torch.int8)
dequantized data:  tensor([[1.2134, 2.3213, 3.3764],
        [4.4843, 5.5921, 6.7000]])
```

非对称映射使用数据max - min的范围进行放缩，同时需要重新计算零点。

scale = r_max - r_min / q_max - q_min 

zero_point = q_min - r_min/scale : 在int8中的位置

x_int8 = X/scale + zero_point

在进行反量化的时候

x_fp32 = (x_int8 - zero_point)*scale


```python3
def quantize(x_fp32):
    qmin = -2 ** (8 - 1)  # 最小值，例如 8-bit 量化的 -128
    qmax = 2 ** (8 - 1) - 1  # 最大值，例如 8-bit 量化的 127

    # 计算输入数据的最小值和最大值
    min_val = torch.min(x_fp32).item()
    max_val = torch.max(x_fp32).item()

    # 计算缩放因子和零点
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - round(min_val / scale)
    x_int8 = torch.round(x_fp32/scale + zero_point)
    x_int8 = torch.clamp(x_int8, qmin, qmax).to(torch.int8)  # 限制在量化范围内

    return x_int8, scale, zero_point

def dequantize(x_int8, scale, zero_point):
    x_fp32 = (x_int8.to(torch.float32) - zero_point) * scale
    return x_fp32
```