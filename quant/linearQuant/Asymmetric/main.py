import torch

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

x_fp32 = torch.tensor([[1.2, 2.3, 3.4],
                       [4.5, 5.6, 6.7]], dtype=torch.float32)

x_int8, scale, zero_point = quantize(x_fp32)
x_outf32 = dequantize(x_int8, scale, zero_point)

print("input data: ", x_fp32)
print("quantized data : ",x_int8)
print("dequantized data: ", x_outf32)
