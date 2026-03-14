#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h> // 必须包含，用于处理 Tensor

namespace nb = nanobind;
using namespace nanobind::literals;
// 简单的加法
int add(int a, int b) {
    return a + b;
}

// 核心示例：原地修改 Tensor
// 参数说明：
// nb::ndarray<
//    float,              // 数据类型
//    nb::ndim<1>,        // 维度 (1维数组)
//    nb::device::cpu,    // 设备 (这里演示 CPU，如果是 GPU 改为 nb::device::cuda)
//    nb::c_contig        // 内存布局 (必须是 C 连续的)
// >
void add_one_inplace(nb::ndarray<float, nb::ndim<1>, nb::device::cpu, nb::c_contig> array) {
    // 1. 获取原始指针
    // nanobind 会自动处理 DLPack 协议，从 PyTorch tensor 拿到指针
    float* ptr = array.data();
    
    // 2. 获取大小
    size_t size = array.shape(0);

    // 3. 执行计算 (原地修改)
    for (size_t i = 0; i < size; ++i) {
        ptr[i] += 1.0f;
    }
}

NB_MODULE(my_ext, m) {
    m.def("add", &add);
    
    // 这里的 "a"_a 是指定参数名为 "a"
    m.def("add_one_inplace", &add_one_inplace, "a"_a.noconvert()); 
    // .noconvert() 表示如果传入的类型不对(比如传了double)，不要隐式复制转换，直接报错。
    // 这在高性能场景下很重要，防止意外的内存拷贝。
}