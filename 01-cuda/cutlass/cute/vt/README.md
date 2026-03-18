# cute 中 _v _t 后缀

在cute中常见的有 `cosize_v`, `rank_v`这类函数， `_v` 后缀的用途是从底层提取出该类型的编译期值

如下所示
```cpp
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    // auto shape = make_shape(1, 2, 3);
    // 在编译器直接计算出结果
    auto shape = make_shape(_1{}, _2{});
    auto shape1 = make_shape(Int<1>{}, Int<2>{});

    // 在运行时计算结果
    auto shape2 = make_shape(1, 2);
    int dim = rank_v<decltype(shape)>;
    printf("dim: %d\n", dim);
}
```

接下来我们来看一下`rank_v`的定义，其实直接去拆包得到value值

```cpp
template <class IntTuple>
static constexpr auto rank_v = rank_t<IntTuple>::value;
```