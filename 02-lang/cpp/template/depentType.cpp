#include <iostream>

// 一个简单的类，内部定义了类型
struct TypeA {
    using InternalType = int;   // InternalType 是一个类型
};

// 另一个类，内部定义了变量
struct VarA {
    static int InternalType;    // InternalType 是一个变量
};
int VarA::InternalType = 10;

// --- 问题的核心：模板函数 ---
template <typename T>
void complex_function() {
    // 报错隐患行：
    // T::InternalType * ptr; 
    
    /* 编译器的困惑：
       1. 如果 T 是 TypeA，这行代码应该是定义一个指针变量：int * ptr;
       2. 如果 T 是 VarA，这行代码应该是一个乘法表达式：VarA::InternalType 乘以 ptr;
       
       为了安全，C++ 规定：在模板中，编译器默认认为 T::Member 是变量。
       如果你想让它代表类型，必须在前面显式加 typename。
    */
    
    typename T::InternalType * ptr; // 明确告诉编译器：这是类型，不是变量
    std::cout << "Successfully treated as a type!" << std::endl;
}

int main() {
    complex_function<TypeA>();
    // complex_function<VarA>(); // 如果取消注释，这行会报错，因为 VarA 里的不是类型
    return 0;
}