import torch

def quantize(x_fp32):
    scale = torch.max(torch.abs(x_fp32))/127.0
    scale += 1e-7
    
    x_int8 = x_fp32/scale
    # 对于正数和负数的情况进行
    x_int8 += 0.5*torch.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(torch.int8)
    return x_int8, scale

x_fp32 = torch.tensor([[1.2, 2.3, 3.4],
                       [4.5, 5.6, 6.7]], dtype=torch.float32)

def dequantize(x_int8, scale):
    x_fp32 = x_int8.to(torch.float32) * scale
    return x_fp32

x_int8, scale = quantize(x_fp32)
x_outf32 = dequantize(x_int8, scale)

print("input data: ", x_fp32)
print("quantized data : ",x_int8)
print("dequantized data: ", x_outf32)