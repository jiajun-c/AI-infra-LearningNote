# cutlass拷贝

cutlass中提供了`copy` 和 `copy_if`的接口用于进行数据的拷贝， 最普通的数据拷贝形式为`copy(src, dst)`，但是这样的拷贝实际上大概率会调用`ld.global.f32` 和 `st.global.f32`，对于我们来说其实际调用的异步/向量/标量是黑盒。

## 1. 基础数据拷贝



## 2. 向量化数据拷贝

```cpp
    // Atom: UniversalCopy<uint128_t> -> 强制使用 128bit 向量指令 (一次搬4个float)
    // Thread Layout: 32x8 (列主序) -> 32行8列的线程阵列
    // Value Layout:  4x1  (列主序) -> 每个线程搬运 4行1列 的数据
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, TA>{}, 
        Layout<Shape<_32, _8>, Stride<_1, _32>>{}, // Thread Layout: M-major (ColMajor)
        Layout<Shape< _4, _1>>{}                   // Value  Layout: M-major (ColMajor)
    );
```