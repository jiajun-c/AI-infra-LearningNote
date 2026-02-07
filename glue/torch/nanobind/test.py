import my_ext
import torch
import numpy as np

# 1. 测试普通函数
print(f"1 + 1 = {my_ext.add(1, 1)}")

# 2. 测试 PyTorch Tensor 交互 (Zero-Copy)
# 创建一个 CPU Tensor
t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

print("Before C++:", t)

# 关键点：直接把 torch tensor 传给 nanobind
# 这一步没有发生数据拷贝，C++ 拿到了 t 的指针
my_ext.add_one_inplace(t)

print("After C++: ", t)

# 验证
expected = torch.tensor([2.0, 3.0, 4.0])
assert torch.allclose(t, expected)
print("PyTorch integration successful!")

# 3. 测试 Numpy (也兼容)
n = np.array([10.0, 20.0], dtype=np.float32)
my_ext.add_one_inplace(n)
print("Numpy result:", n)