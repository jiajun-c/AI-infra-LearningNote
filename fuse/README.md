# 算子融合

## 1. 通过fuse_modules进行融合

如下所示，先声明一个模型，然后调用 `fuse_modules` 对其中的组合进行融合。本质上是通过减少访存的次数来提高效率。

```python3
import torchvision.models as models
from torch.quantization import fuse_modules

model = models.resnet18()
modules_to_fuse = [
    ['conv1', 'bn1', 'relu'],          # 输入层组合
    ['layer1.0.conv1', 'layer1.0.bn1'] # 残差块内组合
]
fused_model = fuse_modules(model, modules_to_fuse)
```
