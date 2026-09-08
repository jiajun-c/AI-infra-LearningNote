#include <iostream>
#include <vector>

using namespace std;

class Widget {
    public:
    string name;
    void print() const {
        printf("widget\n");
    }
    void setName(const string &n) {
        name = n;
    }
};

class Cache {
    mutable int hits = 0;
public:
    void lookup() {
        ++hits;
        printf("hit %d\n", hits);
    }
};

int main() {
    const int x = 10;
    int const y = 10;

    const int* p1 = &x;
    int const* p2 = &x;
    printf("%d %d\n",*p1, *p2);
    Widget w;
    w.print();
    vector<int>v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
        printf("%d\n",*it);
    }
    Cache c;
    c.lookup();
    c.lookup();
}