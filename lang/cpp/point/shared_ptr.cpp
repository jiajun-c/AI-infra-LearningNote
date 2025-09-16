#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int>a = make_shared<int>(5);
    std::cout << "a: " << a.use_count() << "pointer " << a.get() << std::endl;
    printf("%d\n", *a);
    shared_ptr<int>b = a;
    printf("%d\n", *b);
    std::cout << "b: " << a.use_count() << "pointer " << b.get() << std::endl;
}