#include <iostream>
#include <vector>

using namespace std;

class Myarray {
public:
    int *data;
    size_t size = 10;
    Myarray() {
        data = new int[10];
    }
    ~Myarray() {
        delete[] data;
    }
    Myarray(const Myarray& arr) {
        printf("copy construct\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
    }

    Myarray& operator=(const Myarray& arr) {
        printf("copy construct operator\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
        return *this;
    }

    Myarray(Myarray&& arr) {
        printf("move\n");
        data = arr.data;
        arr.data = nullptr;
    }

    Myarray& operator=(Myarray&& arr) {
        data = arr.data;
        arr.data = nullptr;
        return *this;
    }
};


int main() {
    Myarray a;
    Myarray b = std::move(a);
}