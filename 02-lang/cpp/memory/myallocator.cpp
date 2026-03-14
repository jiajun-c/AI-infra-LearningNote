#include <iostream>
#include <vector>
using namespace std;

template <typename T>
struct MyAllocator
{
    // 必需的类型定义
    using value_type = T;

    // 允许分配器在不同类型间转换的构造函数
    template <typename U>
    MyAllocator(const MyAllocator<U>&) noexcept {}

    // // 默认构造函数
    MyAllocator() = default;

    T* allocate(size_t n) {
        std::cout << "allocate " << n << " elements" << std::endl;
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) {
        std::cout << "Deallocating " << n * sizeof(T) << " bytes." << std::endl;
        ::operator delete(p);
    }

    void construct(T* p, const T& val) {
        new(p) T(val);
    }

    void destroy(T* p) {
        p->~T();
    }
};
int main() {
    vector<int, MyAllocator<int>> v;
    v.push_back(1);
    v.push_back(2);
    for (int i = 0; i < v.size(); i++) {
        printf("%d ", v[i]);
    }
    return 0;
}