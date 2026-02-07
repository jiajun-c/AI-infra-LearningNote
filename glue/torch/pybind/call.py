import example_cpp
import numpy as np
import time

# 1. 测试普通函数
print(f"1 + 2 = {example_cpp.add(1, 2)}")

# 2. 测试 Numpy 数组函数
size = 1000000
# 创建 Numpy 数组 (float32 对应 C++ 的 float)
a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
out = np.zeros(size, dtype=np.float32)

print("Starting vector add...")
start = time.time()

# 调用 C++ 函数
example_cpp.vector_add(a, b, out)

end = time.time()
print(f"Vector add finished via C++. Time: {end - start:.6f}s")

# 验证结果
assert np.allclose(out, a + b)
print("Result verification passed!")