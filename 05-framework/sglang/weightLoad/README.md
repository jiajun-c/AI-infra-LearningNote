# 权重加载流程

```python
def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    loader = get_model_loader(load_config, model_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )

```

然后在 `/volume/code/jjcheng/sglang/python/sglang/srt/model_loader/loader.py` 选择 `loader`

对于默认的加载路径

```cpp
        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
```

索引文件，表示每一层的数据是从哪里读取的

```python
{
  "metadata": {
    "total_size": 61064245248
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00016.safetensors",
    "model.layers.0.mlp.experts.0.down_proj.weight": "model-00001-of-00016.safetensors",
  }
}
```

对应的命名
- model: 模型根
- layers: 所有层
- 0: 第0层
- mlp: mlp 模块
- experts 列表
- 0: expert的编号
- down_proj.weight: 权重张量

```shell
model.layers.0.mlp.experts.0.down_proj.weight
│     │       │ │   │       │ └─────────────── 权重张量
│     │       │ │   │       └───────────────── expert 编号（0, 1, 2, ... 127 共128个）
│     │       │ │   └───────────────────────── experts 列表
│     │       │ └───────────────────────────── mlp 模块
│     │       └─────────────────────────────── 第 0 层（transformer layer）
│     └─────────────────────────────────────── 所有层
└───────────────────────────────────────────── 模型根
```

通过`_get_weights_iterator`创建迭代器，然后 `model.load_weights(iterator)` 加载迭代器

```shell
get_model()
  └─ get_model_loader()              # 选择 loader
       └─ loader.load_model()
            ├─ _initialize_model()   # 创建空模型 + 注入量化 config
            ├─ _prepare_weights()    # 发现/下载权重文件（支持 HF Hub / ModelScope）
            ├─ _get_weights_iterator()  # 根据格式创建流式迭代器
            │    ├─ safetensors_weights_iterator()    (内存映射，可多线程)
            │    ├─ fastsafetensors_weights_iterator() (GPU Direct Storage)
            │    ├─ pt_weights_iterator()
            │    └─ gguf_quant_weights_iterator()
            ├─ model.load_weights(iterator)  # 模型侧逐参数加载
            │    └─ 对每个 (name, tensor):
            │         ├─ checkpoint 名 → 模型参数名映射
            │         ├─ 堆叠参数 (q/k/v → qkv_proj, gate/up → gate_up_proj)
            │         └─ 调用 weight_loader 写入参数
            └─ load_weights_and_postprocess()  # 量化后处理
                 └─ 对每个有 quant_method 的模块调用 process_weights_after_loading()

```