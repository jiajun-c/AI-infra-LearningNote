all:
	nvcc test_reduceMax.cu -o test_reduceMax -O3 -arch=sm_80 --std=c++17

run: all
	./test_reduceMax

clean:
	rm -f test_reduceMax
