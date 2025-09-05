#include <iostream>
#include <string>

class A {
public:
    virtual void foo() = 0;
};

class B : public A {
public:
    void foo() override {
        std::cout << "bar" << std::endl;
    }
};


int main() {
    // A a;
    B b;
    // a.foo();
    b.foo();

}