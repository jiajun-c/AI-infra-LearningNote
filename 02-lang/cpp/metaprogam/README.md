# C++ 元编程

## 访存方面

### aligned_storage_t

`std::aligned_storage_t<sizeof(T), alignof(T)>` 可以声明一个大小为T，并按照T的系统对齐大小要求进行对齐

