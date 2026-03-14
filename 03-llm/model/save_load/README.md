# 大模型保存与加载

## 1. 模型保存

使用 `torch.save` 可以将模型数据保存到特定的路径

```python
def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str|os.PathLike|BinaryIO|IO[bytes]):
    saved = {}
    saved["model"] = model.state_dict()
    saved["optimizer"] = optimizer.state_dict()
    saved["iteration"] = iteration
    torch.save(saved, out)
```

### 1.1 保存模型参数（推荐）

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = Model(*args, **kwargs)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 1.2 保存完整模型

```python
# 保存
torch.save(model, 'model_complete.pth')

# 加载
model = torch.load('model_complete.pth')
```

### 1.3 分布式模型保存

对于大模型，通常使用分布式训练，需要特殊处理：

```python
# FSDP 保存
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_config=save_cfg):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, 'model.pth')
```

## 2. 模型加载

使用 `torch.load` 将数据从指定路径加载到变量中，再从变量中加载模型的优化器等信息。

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

### 2.1 安全加载

```python
# PyTorch 2.4+ 建议使用 weights_only=True
model.load_state_dict(
    torch.load('model.pth', weights_only=True, map_location='cuda:0')
)
```

### 2.2 处理不同设备

```python
# 直接加载到指定设备
checkpoint = torch.load('model.pth', map_location='cuda:0')

# 或者加载后手动移动
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint)
model = model.to('cuda:0')
```

## 3. 常见格式

| 格式 | 说明 |
|------|------|
| `.pth` / `.pt` | PyTorch 原生格式 |
| `.safetensors` | 安全的张量格式（推荐） |
| `.ckpt` | 检查点格式 |
| `.onnx` | 跨平台格式 |

## 4. Safetensors 格式

Safetensors 是一种安全的张量序列化格式，避免了 pickle 的安全问题。

```python
from safetensors.torch import save_file, load_file

# 保存
save_file(model.state_dict(), "model.safetensors")

# 加载
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
```

## 5. 大模型加载优化

### 5.1 使用 mmap

```python
# 使用内存映射减少内存占用
checkpoint = torch.load('model.pth', mmap=True)
```

### 5.2 分块加载

对于超大模型，可以分块加载参数：

```python
def load_model_in_chunks(model, checkpoint_path, chunk_size=1000000):
    checkpoint = torch.load(checkpoint_path)
    for name, param in model.named_parameters():
        if name in checkpoint:
            param.data.copy_(checkpoint[name])
```
