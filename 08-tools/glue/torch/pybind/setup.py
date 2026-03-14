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