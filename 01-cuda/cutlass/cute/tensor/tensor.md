# cute Tensor

## 1. Tensor的创建

当我们完成shape和layout的创建后，使用从外部传递的内存地址，可以开始进行tensor的创建，如下所示使用`make_tensor`创建一个global memory 上的tensor对象

```cpp
__global__ void create_tensor_kernel(float* global_ptr) {
    auto layout = make_layout(make_shape(2, 2, 2));
    auto tensor = make_tensor(make_gmem_ptr(global_ptr), layout);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 使用 CuTe 内置的 print 函数
        printf("Layout info:\n");
        print(layout); 
        printf("\n\nTensor info:\n");
        print(tensor);
        printf("\n");
        
        // 演示：访问 Tensor 的第 0 个元素
        // 语法: tensor(坐标)
        printf("Element at index 0: %f\n", tensor(0, 1, 1));
    }
}
```

创建共享内存上的Tensor如下所示，使用`make_smem_ptr`创建一个共享内存上的指针，然后使用`make_tensor`创建一个共享内存上的tensor对象

```cpp
__shared__ float smem[64];
auto smemLayout = make_layout(make_shape(4, 4, 4));
auto tensor_smem = make_tensor(make_smem_ptr(smem), smemLayout);
```

## 2. Tensor的访问

tensor的访问的访问方式有下面的几种
- `[]` 仅一个值
- `()` 可以传递多个值
- `make_coord` 更直观地访问多维度地Tensor

```cpp
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
```

除开固定索引的方式对tensor进行访问，我们可以使用占位符的方式来获取一块的区域，如下所示，每个线程获取到一片长度为VecElem的空间

```cpp
    auto block_shape = make_shape(BlockThreads{}, VecElem{});
    auto block_stride = make_stride(VecElem{}, Int<1>{});
    // 构建 Global Tensor (指针偏移到当前 Block)
    auto gIn  = make_tensor(make_gmem_ptr(d_in  + blk_idx * BLOCK_TILE_SIZE), 
                            make_layout(block_shape, block_stride));
    auto gOut = make_tensor(make_gmem_ptr(d_out + blk_idx * BLOCK_TILE_SIZE), 
                            make_layout(block_shape, block_stride));

    auto tIgIn  = gIn(threadIdx.x, _);
    auto tOgOut = gOut(threadIdx.x, _);
```

## 3. Tensor的具体结构

Tensor的本质是Engine和layout组成的，Engine负责数据的存储，Layout负责数据的布局
如下所示，其最终由一个会落到一个rep_中

```cpp
template <class Engine, class Layout>
struct Tensor
{
  using iterator     = typename Engine::iterator;
  using value_type   = typename Engine::value_type;
  using element_type = typename Engine::element_type;
  using reference    = typename Engine::reference;

  using engine_type  = Engine;
  using layout_type  = Layout;
  // ...
  cute::tuple<layout_type, engine_type> rep_;
}
```

cutlass中的Engine可以分为两类
- ViewEngine：并不持有数据，值保存指针
- ArrayEngine：保存实际的数据，使用cuda中的寄存器进行存储
