import torch


class Exp(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, _ = ctx.saved_tensors
        return grad_output * result
    
input = torch.tensor([1.0, 2.0, 3.0])
print(Exp.apply(input))