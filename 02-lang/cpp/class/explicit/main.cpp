#include <iostream>
using namespace std;

class Points {
public:
    int x, y;
    explicit  Points(int x = 0, int y = 0) {
        this->x = x;
        this->y = y;
    }
};

void displayPoint(const Points& p) 
{
    cout << "(" << p.x << "," 
         << p.y << ")" << endl;
}

int main()
{
    displayPoint(Points(1));
    printf("now\n");
    Points p = 1;
}