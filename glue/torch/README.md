# torch 自定义cuda 算子

下面将介绍一下torch中自定义cuda算子的几种方式

## 1. 借助dsl

现在已经有很多的dsl，如triton，cute-DSL，tilelang等，这些dsl的好处是可以直接写python代码

## 2. Pybind/nanobind

pybind 是最经典的python调用cpp代码方式之一,其支持C++11及以上，而nanobind仅支持C++17及以上

如下所示是一个C++的函数，我们将其绑定到一个python函数中

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // 必须包含这个头文件来支持 Numpy 交互

namespace py = pybind11;

// 1. 简单的 C++ 函数
int add(int i, int j) {
    return i + j;
}

// 2. 高效的 Numpy 数组操作 (C++ 修改 Python 传入的数组)
// 功能：实现 out = a + b
void vector_add(py::array_t<float> a, py::array_t<float> b, py::array_t<float> out) {
    // 获取数组的 buffer 信息
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();
    py::buffer_info buf_out = out.request();

    // 简单的形状检查
    if (buf_a.size != buf_b.size || buf_a.size != buf_out.size) {
        throw std::runtime_error("Input shapes must match");
    }

    // 获取数据指针 (这里是 C++ 原始指针，性能最高)
    float *ptr_a = static_cast<float *>(buf_a.ptr);
    float *ptr_b = static_cast<float *>(buf_b.ptr);
    float *ptr_out = static_cast<float *>(buf_out.ptr);

    // 执行计算 (如果是 CUDA 开发，这里就是 launch kernel 的地方)
    for (size_t i = 0; i < buf_a.size; i++) {
        ptr_out[i] = ptr_a[i] + ptr_b[i];
    }
}

// 3. 绑定模块
// PYBIND11_MODULE(模块名, 变量名)
PYBIND11_MODULE(example_cpp, m) {
    m.doc() = "pybind11 example plugin"; // 模块文档

    // 绑定普通函数
    m.def("add", &add, "A function that adds two numbers",
          py::arg("i"), py::arg("j")); // 指定参数名

    // 绑定 Numpy 函数
    m.def("vector_add", &vector_add, "Vector addition using raw pointers");
}
```
如下所示是setup.py中的内容，编译我们上面的C++代码

```python3
from setuptools import setup, Extension
import pybind11

# 定义扩展模块
ext_modules = [
    Extension(
        "example_cpp",              # 编译生成的模块名 (import example_cpp)
        ["example.cpp"],            # C++ 源文件路径
        include_dirs=[pybind11.get_include()],  # 自动找到 pybind11 头文件
        language='c++',
        extra_compile_args=['-std=c++11'],      # 指定 C++ 标准
    ),
]

setup(
    name="example_cpp",
    version="0.0.1",
    author="User",
    description="A simple pybind11 example",
    ext_modules=ext_modules,
)
```

编译完成后我们就可以直接在py侧调用该函数了

## 3. torch C++ extenstion



## 4. cppyy

## 5. Cpython

## 6. ctypes
