import torch
from torch.utils.flop_counter import FlopCounterMode
from typing import Union, Tuple

def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

from torchvision.models import resnet18

model = resnet18()

print("forward: ", get_flops(model, (1, 3, 224, 224)))
print("forward+backward:", get_flops(model, (1, 3, 224, 224), with_backward=True))