#include <iostream>
#include <bitset>

using namespace std;

int main()
{
    bitset<128>bitset1; 
    bitset<128>bitset2;
    
    bitset2.set(1, 1);
    printf("the bitset2 count: %d\n", bitset2.count());
    bitset1[0] = 1;
    printf("the bitset1 count: %d\n", bitset1.count());
    printf("the bitset1 has one: %d\n", bitset1.any());
    printf("the bitset1 has zero: %d\n", bitset1.none());
}