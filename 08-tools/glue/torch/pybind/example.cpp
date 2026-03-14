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