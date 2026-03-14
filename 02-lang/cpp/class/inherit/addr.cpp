#include <iostream>

class A { int a; };
class B { int b; };
class C : public A, public B { public: int c; };

int main() {
    C obj;
    C* ptrC = &obj;

    // 1. 转为第一个基类
    A* ptrA = static_cast<A*>(ptrC);
    
    // 2. 转为第二个基类 (static_cast 会调整偏移)
    B* ptrB = static_cast<B*>(ptrC);
    
    // 3. 错误的转换 (reinterpret_cast 不调整偏移)
    B* ptrB_Wrong = reinterpret_cast<B*>(ptrC);

    std::cout << "C ptr:      " << &obj.c << std::endl;
    std::cout << "C ptr:      " << ptrC << std::endl;
    std::cout << "A ptr:      " << ptrA << " (Offset: " << (long)ptrA - (long)ptrC << ")" << std::endl;
    std::cout << "B ptr:      " << ptrB << " (Offset: " << (long)ptrB - (long)ptrC << ")" << " <--- static_cast Correct!" << std::endl;
    std::cout << "B ptr(bad): " << ptrB_Wrong << " (Offset: " << (long)ptrB_Wrong - (long)ptrC << ")" << " <--- reinterpret_cast Wrong!" << std::endl;

    return 0;
}