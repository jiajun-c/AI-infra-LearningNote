# 大模型保存加载

## 1. 模型保存

使用`torch.save`可以将模型数据保存到特定的路径

```python
def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str|os.PathLike|BinaryIO|IO[bytes]):
    saved = {}
    saved["model"] = model.state_dict()
    saved["optimizer"] = optimizer.state_dict()
    saved["iteration"] = iteration
    torch.save(saved, out)

```

## 2. 模型加载

使用`torch.load`将数据从指定路径加载到变量中，再从变量中加载模型的优化器等信息。

```python
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer):
    saved = torch.load(src)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    return saved["iteration"]
```