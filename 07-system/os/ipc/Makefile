CC ?= gcc
CFLAGS ?= -Wall -Wextra -O2 -std=c11

.PHONY: all clean run

all: pipe_features_demo

pipe_features_demo: pipe_features_demo.c
	$(CC) $(CFLAGS) -o $@ $<

run: pipe_features_demo
	./pipe_features_demo

clean:
	rm -f pipe_features_demo
