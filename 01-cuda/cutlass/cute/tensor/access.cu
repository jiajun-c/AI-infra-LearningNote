#include "cute/tensor_impl.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // 1. 定义一个 Layout
    // 形状: (4, 8)
    // 步长: (1, 4) -> Column-Major (列主序)，即 LayoutLeft
    auto layout = make_layout(make_shape(4, 8), LayoutLeft{});
    
    // 2. 打印 Layout 信息
    print("Layout: "); print(layout); print("\n");
    // 输出: (_4,_8):(_1,_4)

    // 3. 模拟一段内存 (在 CPU 栈上)
    int *data = new int[32];
    for(int i = 0; i < 32; ++i) data[i] = i;

    // 4. 创建 Tensor
    auto tensor = make_tensor(data, layout);

    // --- 演示三种访问 ---

    // 目标：访问第 2 行，第 1 列的元素
    // 在列主序中：Offset = row + col * stride_row = 2 + 1 * 4 = 6
    int row = 2;
    int col = 1;

    // 方式 A: [] 线性索引
    // 我们必须自己知道 offset 是 6
    int val_linear = tensor[6]; 
    std::cout << "Access via []: " << val_linear << std::endl;

    // 方式 B: () 多维坐标
    // 最常用的方式，直观
    int val_multi = tensor(row, col);
    std::cout << "Access via (): " << val_multi << std::endl;

    // 方式 C: make_coord
    // 显式打包坐标
    auto coord = make_coord(row, col);
    int val_coord = tensor(coord);
    std::cout << "Access via make_coord: " << val_coord << std::endl;

    // --- 进阶：切片 (Slicing) ---
    // Tensor 的 () 不仅可以返回引用，还可以返回子 Tensor (Slice)
    
    // 取第 1 列的所有元素
    // _ 类似于 Python 中的 : (冒号)
    auto col_1_tensor = tensor(_, 1); 
    print("Column 1 Slice: "); print(col_1_tensor); print("\n");
    
    return 0;
}