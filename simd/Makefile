simd: simd.c
	clang $^ -Wall -Wextra -o $@

.PHONY: test clean

run: simd
	./simd

clean: simd
	rm simd
