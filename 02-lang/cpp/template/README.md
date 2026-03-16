# C++ 模板

## 全特化

C++ 可以通过对模板进行特化，通过传入不同的类型参数来决定调用什么函数，如下所示，定义一个枚举类型，然后为几个特化的方式设计函数，如下所示

```cpp
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;



template <CalMode mode>
struct Calculator;

template <>
struct Calculator<CalMode::SIMPLE_INT> {
    static void compute(int a, int b) {
        std::cout << "[Simple Mode] Using Integer Math:" << std::endl;
        std::cout << "  " << a << " + " << b << " = " << (a + b) << std::endl;
    }
    static constexpr int ID = 1;
};

template <>
struct Calculator<CalMode::ADVANCED_FLOAT> {
    static void compute(float a, float b) {
        std::cout << "[Advanced Mode] Using Complex Math:" << std::endl;
        float result = std::sqrt(a * a + b * b);
        std::cout << "  Hypotenuse(" << a << ", " << b << ") = " << result << std::endl;
    }

    static void special_hello() {
        std::cout << "  -> Only Advanced Mode has this function!" << std::endl;
    }
};

int main() {
    // 场景 1：使用简单模式
    // 编译器看到 <CalcMode::SIMPLE_INT>，自动去寻找特化版本 A
    Calculator<CalMode::SIMPLE_INT>::compute(3, 4);

    std::cout << "-------------------" << std::endl;

    // 场景 2：使用高级模式
    // 编译器看到 <CalcMode::ADVANCED_FLOAT>，自动去寻找特化版本 B
    Calculator<CalMode::ADVANCED_FLOAT>::compute(3.0f, 4.0f);
    
    // 调用特有的函数
    Calculator<CalMode::ADVANCED_FLOAT>::special_hello();

    // 下面这行如果解开注释会报错，因为 SIMPLE_INT 模式的结构体里没有这个函数
    // Calculator<CalcMode::SIMPLE_INT>::special_hello(); 

    return 0;
}
```

## TAD(模板推导)

C++的模板可以根据输入的类型来做一个类型的推导，但是如何输入中没有变量可以用于推导，那么将会无法进行类型的推导

```cpp
#include <iostream>

template <typename T>
void foo(T x) {
    std::cout << x << std::endl;
}

int main() {
    foo(1.1);
    foo(1);
}
```

## Dependent Types

模版依赖指的是一个类型依赖于模板参数，编译器在解析代码的时候，无法预先知道这个名字到底是一个类型还是一个变量

```cpp
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
```