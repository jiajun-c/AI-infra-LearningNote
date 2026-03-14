from collections.abc import Sequence

print(isinstance([1, 2, 3], Sequence))    # True
print(isinstance((1, 2, 3), Sequence))    # True
print(isinstance("hello", Sequence))      # True
print(isinstance(range(5), Sequence))     # True（range 也实现了协议）