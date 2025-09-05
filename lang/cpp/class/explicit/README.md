# Explicit 关键词

通过Explicit关键词可以防止出现隐式调用构造函数，

如下所示`Points p = 1;`无法自动调用构造函数

```cpp
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
    Points p = 1;
}
```
