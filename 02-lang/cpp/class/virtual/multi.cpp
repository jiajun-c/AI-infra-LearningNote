#include <iostream>
#include <vector>
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

class bar
{
private:
    size_t b = 43;

public:
    virtual void fun4()  {std::cout << "bar::fun4" << std::endl;}
    virtual void fun5()  {std::cout << "bar::fun5" << std::endl;}
};


class far
{
private:
    size_t c = 44;

public:
    virtual void fun6() {std::cout << "far::fun6" << std::endl;}
    virtual void fun7() {std::cout << "far::fun7" << std::endl;}
};

class Goo: public foo, public bar, public far
{
private:
    size_t d = 45;

public:
virtual void fun2() override {std::cout <<"Goo::fun2" << std::endl;}

virtual void fun6() override {std::cout <<"Goo::fun6" << std::endl;}
};
using PF = void(*)();

void test(Goo *pf) {
    size_t* virtual_point = (size_t*)pf;
    PF* pf1 = (PF*)*virtual_point;
    PF* pf2 = pf1 + 1;
    PF* pf3 = pf1 + 2;
    PF* pf4 = pf1 + 3;

    (*pf1)();
    (*pf2)();
    (*pf3)();
    (*pf4)();
}
void test1(Goo *pf) {
    size_t* virtual_point = (size_t*)pf + 2;
    PF* pf1 = (PF*)*virtual_point;
    PF* pf2 = pf1 + 1;

    (*pf1)();
    (*pf2)();
}

void test2(Goo *pf) {
    size_t* virtual_point = (size_t*)pf + 4;
    PF* pf1 = (PF*)*virtual_point;
    PF* pf2 = pf1 + 1;

    (*pf1)();
    (*pf2)();
}

int main() { 
    // Goo g;
    // g.fun1();
    // g.fun2();
    // g.fun3();
    // g.fun4();
    // g.fun5();
    // g.fun6();
    Goo *gp = new Goo();
    test(gp);
    test1(gp);
    test2(gp);
    size_t* virtual_point = (size_t*)gp;
    size_t* ap = virtual_point + 1;
    size_t* bp = virtual_point + 3;
    size_t* cp = virtual_point + 5;
    size_t* dp = virtual_point + 6;
    std::cout << *ap << std::endl;  //42
    std::cout << *bp << std::endl;  //1024
    std::cout << *cp << std::endl;  //42
    std::cout << *dp << std::endl;  //1024
}