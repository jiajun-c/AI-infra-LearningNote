# CuTe layout

cute的layout将逻辑坐标和物理内存偏移完成了解耦

一个layout对象由两个部分组成
1. shape，定义了逻辑空间的维度
2. stride，定义了在这个维度上移动一步将会跳过多少元素

如下所示是一个一维静态形状的Tensor，其形状为8，stride为1
```cpp
auto s8 = make_layout(Int<8>{});    
print("s8: "); print(s8); print("\n");
// 动态形状 (Dynamic Shape)，运行时决定，由寄存器存值
// Layout<int, Int<1>>
auto d8 = make_layout(8);
```

## 1. 静态形状和动态形状

只有当cute中认为传入的是字面量的时候，才会认定其是静态形状

如下所示的两种写法其实都可以被称为静态shape

```shell
auto s8 = make_layout(_8{});    
auto s8 = make_layout(Int<8>{});
```

而动态shape则是传入一个变量或者左值

```shell
 auto d8 = make_layout(n);
```

打印两种输出

```shell
s8: _8:_1
d8: 8:_1
```

如果不指定stride，那么将会根据列主序的逻辑进行来计算stride，只有当传入LayoutRight的参数的时候才会按照行主序进行计算，如下所示，默认是LayoutLeft(列主序)

```cpp
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/stride.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;
// using namespace 
int main() {
    // cuda
    auto shape = make_shape(2, 4);
    auto layoutRow = make_layout(shape, LayoutRight{});
    auto layoutCol = make_layout(shape, LayoutLeft{});
    print(layoutRow);
    print(layoutCol);
}
//  (2,4):(4,_1)(2,4):(_1,2)
```

## 2. 对于layout的操作

使用make_layout两个layout合并为一个layout
使用append可以在layout后面增加一个维度，使用prepend可以在layyout前面增加一个维度

```cpp
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    auto shape1 = make_shape(1, 3);
    auto shape2 = make_shape(2, 4);
    auto layout1 = make_layout(shape1);
    auto layout2 = make_layout(shape2);
    auto layoutA = make_layout(layout1, layout2);
    auto layoutB = append(layout1, make_layout(Int<4>{}));
    auto layoutC = prepend(layout1, make_layout(Int<5>{}));
    print(layout1);print("\n");
    print(layout2);print("\n");
    print(layoutA);print("\n");
    print(layoutB);print("\n");
    print(layoutC);print("\n");

}
```


进行分组

```cpp
    auto layout4dims = make_layout(make_shape(1, 2, 3, 4));
    auto groupLayout = group<0, 2>(layout4dims);
    print(groupLayout);
```

对维度信息进行递归清理

```cpp
    auto a = make_layout(Shape <_2, Shape <_1, _6>>{},
                           Stride<_1, Stride<_6,_2>>{});

    print(a);print("\n");

    auto ca = coalesce(a, Step<_1,_1>{});
    print(ca);print("\n");
```

获取layout/shape的维度的数量，`rank<int>()` 可以获取到某个维度的维度数量，例如如下所示的((4, 4), 2, 3)，其第一个内嵌维度的维度数量为2，其完整的内嵌维度数量为3, 非内嵌维度对应的维度数量为1 

```cpp
#include "cute/layout.hpp"
#include <cute/tensor.hpp>
#include <iostream>

// using namespace std;
using namespace cute;
int main() {
    auto shape = make_shape(make_shape(4, 4), 2, 3);
    auto layout = make_layout(shape);

    print(rank(layout)); print("\n");
    print(rank<0>(layout)); print("\n");
    print(rank<1>(layout)); print("\n");
}
```

获取到某个维度的大小，支持嵌套查询，比如 size<0, 1> 表示获取到第一个维度内的第一个维度

```cpp
#include "cute/layout.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(make_shape(4, 5), 6);
    auto layout = make_layout(shape);

    print(size<0, 0>(layout));print("\n");
}
```

对shape/layout 进行拆分，如下所示使用get获取到某个shape/layout

```cpp
#include "cute/layout.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), Int<4>{});
    auto s0 = get<0>(shape);
    print(s0); print("\n");

    auto s1 = get<1>(shape);
    print(s1); print("\n");
}

```

## 3. 函数复合

函数复合操作使得我们可以以一种设定的视角去访问原先的数据，例如对矩阵进行reshape或者是转置

reshape的例子，将长度为10的线性数据组织为2x5的列主序矩阵

```cpp
#include "cute/layout.hpp"
#include <iostream>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // 1. 物理现实 (A): 一个 10个元素 的线性数组
    // Index: 0 1 2 3 4 5 6 7 8 9
    Layout A = make_layout(Int<10>{}, Int<1>{});

    // 2. 逻辑视角 (B): 我想把它看成一个 2x5 的矩阵 (列主序)
    // 意思就是: 
    // Col 0: 0, 1
    // Col 1: 2, 3 ...
    // Shape: (2, 5), Stride: (1, 2)
    Layout B = make_layout(make_shape(_2{}, _5{}), make_stride(_1{}, _2{}));

    // 3. 组合 (R = A o B)
    // R 现在就是一个 "虚拟" 的 2D 矩阵
    auto R = composition(A, B);

    print("Layout R: "); print(R); print("\n");
    // 输出: (_2,_5):(_1,_2)

    // 4. 验证
    // 我访问 R 的 (1, 2) -> 第 2 列 第 1 行
    // 逻辑上: B(1, 2) = 1*1 + 2*2 = 5 (它是第 5 个元素)
    // 物理上: A(5) = 5
    // 结果: 5
    print(R(1, 2)); 
}
```

矩阵转置，在上文中得到的B矩阵的基础上，就是将stride的顺序进行交换，将列主序的stride变为行主序的stride，打印得到结果就是行主序的顺序了

```cpp
    Layout BT = make_layout(make_shape(_2{}, _5{}), make_stride(_5{}, _1{}));
    auto RT = composition(B, BT);
    print(RT(1, 2)); print("\n");
```


## 4. 补集

补集操作指的是给定一个低维空间的描述，然后补充一个高维空间的描述，使得其可以根据高纬空间的补充来描述完整的空间

```cpp
#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/util/print.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // 1. 定义全集 (The Whole)
    // 假设是一个 128 大小的空间
    auto shape_M = Int<128>{};

    // 2. 定义子集 (The Part / Tile)
    // 我们取前 32 个元素，步长为 1
    Layout layout_A = make_layout(Int<32>{}, Int<1>{});

    // 3. 计算补集 (The Rest)
    // 问：为了填满 128，我还需要怎么走？
    auto layout_R = complement(layout_A, shape_M);

    std::cout << "Whole: " << shape_M << std::endl;
    std::cout << "Part:  " << layout_A << std::endl;
    std::cout << "Rest:  " << layout_R << std::endl;
    print(layout_A(0));print("\n");
    print(layout_R(0));print("\n");
    // 预期输出 Rest: (_4):_32
    // 解释：
    // _4  -> 还需要 4 步 (因为 32 * 4 = 128)
    // _32 -> 每步跨度 32 (因为 Part 已经占了 32)
    
    
    // 4. 进阶：二维场景
    // 假设全集是 (16, 16) 列主序，也就是 Stride (1, 16)
    // 我们选了一列 (16, 1)，Stride (1, 0) <-- 注意这里的0，意味着不跨列
    // complement 会告诉我们如何跨列
    auto shape_N = make_shape(Int<16>{}, Int<16>{});
    Layout layout_B = make_layout(make_shape(_16{}, _1{}), make_stride(_1{}, _0{}));
    auto layout_BR = complement(layout_B, shape_N);
    print(layout_BR);
    return 0;
}
```

## 5. 对Tensor进行分区

使用ziptile的方式可以对矩阵创建一个二级的分区，适用于在大块的矩阵内划分小的分块。

如下所示，将一个(8,24)的列主序的矩阵按照(4, 8)的分块大小进行分块，如下所示，后续可以访问第(0, 0)个块的第(0, 1)个元素，也就是8

```cpp
#include "cute/layout_composed.hpp"
#include "cute/tensor_impl.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {  
    auto layout = make_layout(make_shape(8, 24), LayoutLeft{});
    auto tiler = Shape<_4, _8>{};
    int *data = new int[8*24];
    for (int i = 0; i < 8*24; i++) data[i] = i;
    Tensor a = make_tensor(data, layout);
    Tensor tile_a = zipped_divide(a, tiler);
    auto element= tile_a(make_coord(0, 1), make_coord(0, 0));
    print(element);
    return 0;
}

// 输出为8
```

这样写相对繁琐，我们也可以使用`local_tile`来进行优化写法

如下所示获取到竖向第二个tile
```cpp
    auto tile_b = local_tile(a, tiler, make_coord(1, 0));

    print(tile_b(0, 0));print("\n");
```

也可以使用`logical_divide`进行划分，假设输入是12,分块数量3，分块形状为`(3, 4)`，假设不想对某一个维度做切分，那么可以用`_`来占位

```cpp
#include "cute/layout.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_zip.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    auto layout = make_layout(Int<12>{});
    auto divide_1d = logical_divide(layout, Shape<Int<3>>{});
    print(divide_1d);print("\n");

    auto layout2d = make_layout(make_shape(Int<16>{}, Int<8>{}), GenRowMajor{});
    auto divide2d = logical_divide(layout2d, make_shape(_, Int<4>{}));

    print(divide2d);print("\n");
}
```