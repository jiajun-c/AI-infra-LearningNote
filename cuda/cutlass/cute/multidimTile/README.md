# cute多维度分块

## 1. 基础分块

我们先来看cute中一个最基础的分块，如下所示采用(2, 7)的分块大小对(8, 35)大小的tensor进行分块

```cpp
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    float* a = 0;
    auto shape = make_shape(8, 35);
    auto layout = make_layout(shape);
    auto tensor = make_tensor(a, layout);
    auto tiler = make_shape(2, 7);
    auto cood = make_coord(0, 0);
    auto local = local_tile(tensor, tiler, cood);
    print(local);
}
```

## 2. 高维度分块

使用高维度分块，如下所示，Step忽略中间的维度，使用（3,6）的shape进行分块，然后通过前面适配上的dim维度，后面的维度再通过idx来读取对应的分块形状

```cpp
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    auto shape = make_shape(12, 30);
    auto layout = make_layout(shape);
    float *ptr = 0;
    auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
    auto tiler = make_shape(3, 5, 6);
    auto cood = make_coord(1, _, _);
    // auto cood = make_coord(1, any, _);
    auto local = local_tile(tensor, tiler, cood, Step<_1, X, _1>{});
    print(local(_, _, 0));
}
```