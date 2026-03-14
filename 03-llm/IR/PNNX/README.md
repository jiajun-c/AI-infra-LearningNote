# PNNX 

PNNX 格式是pytorch模型导出的一个中间表示，其相比于ONNX对于torch算子支持更加全面同时计算图等更加简洁。

安装`PNNX`

```shell
pip3 install pnnx
```

## 1.PNNX 格式

## 2.PNNX 使用

先将模型导出为TorchScript格式

```python3
import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()

x = torch.rand(1, 3, 224, 224)

mod = torch.jit.trace(net, x)

mod.save("resnet18.pt")
```

使用pnnx将其转换为pt格式
```shell
pnnx resnet18.pt "inputshape=[1,3,224,224]"
```


