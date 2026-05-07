# 静态成员

静态成员在头文件中声明，但是没有被实例化，只有在cpp文件中才会被实例化

```cpp
// 头文件中声明（只是声明，不分配内存）
class FTensorAllocator {
    static std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>> g_allocators_;
};

// .cpp 中定义（真正分配内存，必须写 ClassName::）
std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>>
    FTensorAllocator::g_allocators_;
```