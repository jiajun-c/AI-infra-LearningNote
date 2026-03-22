import torch
torch._logging.set_logs(graph_code=True)

def foo(x, y):
    a = torch.sin(x)
    b = torch.sin(y)
    return x + y

opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(3, 3), torch.randn(3, 3)))


@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


print(opt_foo2(torch.randn(3, 3), torch.randn(3, 3)))