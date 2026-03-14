import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()

x = torch.rand(1, 3, 224, 224)

mod = torch.jit.trace(net, x)

mod.save("resnet18.pt")