#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

enum class CalMode {
    SIMPLE_INT,
    ADVANCED_FLOAT
};

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