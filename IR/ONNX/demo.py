import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()

x = torch.rand(1, 3, 224, 224)

torch.onnx.export(net, x, "net", input_names=['input'], output_names=['output'])