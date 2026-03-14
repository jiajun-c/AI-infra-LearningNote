#include <iostream>

using namespace std;
class foo
{
private:
    size_t a = 42;
public:
    virtual void fun1() {std::cout << "foo::fun1" << std::endl;}
    virtual void fun2() {std::cout << "foo::fun2" << std::endl;}
    virtual void fun3() {std::cout << "foo::fun3" << std::endl;}
};

class bar : public foo
{
private:
    size_t b = 43;
    size_t c = 44;

public:
    void fun1() override {std::cout << "bar::fun1" << std::endl;}
    void fun3() override {std::cout << "bar::fun3" << std::endl;}
};

using PF = void(*)();

void test(foo *pf) {
    size_t* virtual_point = (size_t*)pf;
    PF* pf1 = (PF*)*virtual_point;
    PF* pf2 = pf1 + 1;
    PF* pf3 = pf1 + 2;
    (*pf1)();
    (*pf2)();
    (*pf3)();
}
int main() {
    foo* fp = new foo();
    test(fp);
    fp = new bar();
    test(fp);
    size_t* virtual_point = (size_t*)fp;
    size_t* ap = virtual_point + 1;
    size_t* bp = virtual_point + 2;
    std::cout << *ap << std::endl;  //42
    std::cout << *bp << std::endl;  //1024
    std::cout << *(bp+1) << std::endl;  //1024


}