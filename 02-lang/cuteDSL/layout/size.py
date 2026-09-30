import cutlass.cute as cute


@cute.jit
def main():
    layout = cute.make_layout((2, 3, 4, 5))
    print("layout =", layout)
    print("总大小 = ", cute.size(layout))
    print("V 维 = ", cute.size(layout, mode=[0]))
    print("N 维 = ", cute.size(layout, mode=[2]))


main()
