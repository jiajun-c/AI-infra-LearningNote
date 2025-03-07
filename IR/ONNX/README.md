# ONNX

ONNX是微软和facebook提出的一种开放格式，可以将各种框架的模型统一导出为ONNX这种统一的权重。

## 1. ONNX 格式


## 2. ONNX 使用

使用onnx将resnet模型导出为onnx格式。

```python3
import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()
 
x = torch.rand(1, 3, 224, 224)

torch.onnx.export(net, x, "net", input_names=['input'], output_names=['output'])
```

## 3. ONNX优化

onnx-simplifer可以对计算图进行优化，


