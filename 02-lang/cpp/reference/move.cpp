#include <iostream>
#include <type_traits> // 仅用于演示测试，手写时不依赖它

// ---------------------------------------------------------
// Step 1: 手写 remove_reference
// 作用：剥离类型的引用属性，还原出“裸类型”
// ---------------------------------------------------------
template <typename T>
struct my_remove_reference {
    using type = T;
};

// 特化版本：处理左值引用 (int&)
template <typename T>
struct my_remove_reference<T&> {
    using type = T;
};

// 特化版本：处理右值引用 (int&&)
template <typename T>
struct my_remove_reference<T&&> {
    using type = T;
};

// 辅助别名 (C++14 风格)，方便使用
template <typename T>
using my_remove_reference_t = typename my_remove_reference<T>::type;


// ---------------------------------------------------------
// Step 2: 手写 move
// 核心逻辑：无论你传进什么(T&& t)，我都强制转成 (裸类型 + &&)
// ---------------------------------------------------------
template <typename T>
constexpr my_remove_reference_t<T>&& my_move(T&& t) noexcept {
    // static_cast 也可以转引用，这里就是强转为右值引用
    return static_cast<my_remove_reference_t<T>&&>(t);
}

template <typename T>
constexpr typename std::remove_reference<T>::type&& my_move1(T &&t) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(t);
}


template<typename T>
constexpr typename std::remove_reference<T>::type && my_move2(T &&t) noexcept {
    return static_cast<typename std::remove_reference<T>::type>(t);
}
// ---------------------------------------------------------
// 测试代码
// ---------------------------------------------------------
class A {
public:
    A() { std::cout << "Construct\n"; }
    A(const A&) { std::cout << "Copy Construct\n"; }
    A(A&&) noexcept { std::cout << "Move Construct\n"; }
};

int main() {
    A a;
    std::cout << "--- Try my_move ---\n";
    // a 是左值，my_move(a) 将其强转为右值，触发 Move Construct
    A b = my_move1(a); 
    return 0;
}