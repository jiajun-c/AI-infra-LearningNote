import torch
from torch.quantization import fuse_modules
import torchvision

# 示例：融合ResNet中的Conv-BN-ReLU
model = torchvision.models.resnet18()

print(model)
modules_to_fuse = [
    ['conv1', 'bn1', 'relu'],          # 第一层组合
    ['layer1.0.conv1', 'layer1.0.bn1'] # 残差块内组合
]
fused_model = torch.ao.quantization.fuse_modules_qat(model, modules_to_fuse)