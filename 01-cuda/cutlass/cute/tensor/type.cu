#include <cstdio>
#include <cute/tensor.hpp>
#include <typeinfo>

// 定义一个打印类型的辅助函数
template <typename T>
__host__ __device__ void print_type_name() {
#if defined(__PRETTY_FUNCTION__)
    printf("Type is: %s\n", __PRETTY_FUNCTION__);
#elif defined(__FUNCSIG__)
    printf("Type is: %s\n", __FUNCSIG__);
#else
    printf("Type name not supported on this compiler.\n");
#endif
}

// 使用 LayoutA_Padded 定义
using namespace cute;
using LayoutA_Padded = Layout<Shape<Int<4>, Int<4>>, Stride<Int<1>, Int<5>>>;

int main() {
    // 1. 获取 cosize_v 的类型
    using TheType = decltype(cosize_v<LayoutA_Padded>);
    int x = 10;
    std::cout << "size type " << typeid((size(LayoutA_Padded{}))).name() << std::endl;
    std::cout << "x type: " << typeid(int(x)).name() << std::endl;
    // 2. 打印它
    print_type_name<TheType>();

    return 0;
}