#include <iostream>

int main()
{
    int *arr = (int *)malloc(sizeof(int) * 10);

    free(arr);
    return 0;
}