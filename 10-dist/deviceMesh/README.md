# torch DeviceMesh

在deviceMesh出现之前，如果需要组合多种并行策略，需要手动创建和管理多个ProcessGroup，计算每个rank输入哪些组。deviceMesh封装了这些计算，用于自动创建通信组

## 1. API

初始化

```python
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 4),
    mesh_dim_names=("dp", "tp"),
)
```

获取NccLGroup

```python
tp_group = mesh.get_group("tp")
dp_group = mesh.get_group("dp")
```

然后就可以复用nccl group的相关API

## 2. 