import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

# 1. 加载配置
model_id = "/volume/code/hisys/models/zai-org/GLM-5/snapshots/83d08ca96c5049150ffb89dd39343746b179e760"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# 2. 在 meta 设备上构建模型图
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 3. 统计唯一的线性层 Shape
unique_shapes = {}  # key: (out_features, in_features), value: 出现的层名称示例

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        w_shape = module.weight.shape
        # 过滤非 2D 矩阵或极小维度
        if len(w_shape) == 2 and w_shape[0] > 1 and w_shape[1] > 1:
            shape_key = (w_shape[0], w_shape[1])
            if shape_key not in unique_shapes:
                unique_shapes[shape_key] = name

# 4. 打印去重后的结果
print(f"{'Weight Shape (N, K)':<25} | {'Count':<8} | {'Example Layer Name'}")
print("-" * 80)

# 按计算量 (N * K) 从大到小排序，方便识别核心算子
sorted_shapes = sorted(unique_shapes.keys(), key=lambda x: x[0] * x[1], reverse=True)

for shape in sorted_shapes:
    # 统计该 shape 出现的次数
    occurrence = sum(1 for m in model.modules() if isinstance(m, nn.Linear) and tuple(m.weight.shape) == shape)
    
    shape_str = f"({shape[0]}, {shape[1]})"
    print(f"{shape_str:<25} | {occurrence:<8} | {unique_shapes[shape]}")

print("-" * 80)
print(f"Total Unique GEMM Shapes: {len(unique_shapes)}")