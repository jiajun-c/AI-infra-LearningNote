# TensorRT

TensorRT 是英伟达推出的高性能深度学习推理库

TensorRT的基本工作流
- 导出AI模型
- 选择精度
- 转换模型
- 进行模型部署

## 1. 模型的导出

模型导出可以将其导出为常见的ONNX或者PNNX格式

## 2. 精度选择

推理通常相比于训练需要更少的精度，选择低精度可以获得更快的计算速度
TensorRT支持FP32，FP16，FP8，BF16，FP8，INT64， INT8和INT4多种精度。

TensorRT中由两套系统
- 弱类型：让TensorRT自己选择如何减少精度1
- 强类型：固定某种类型让TensorRT基于它进行推理

```python3
import numpy as np
PRECISION = np.float32
```

## 3. 模型转换

为了让模型可以通过TensorRT进行推理，我们还需要将其转换为TensoRT Engine格式。


## 4. 模型部署

基于上面转换后的模型，可以使用TensorRT进行运行。


