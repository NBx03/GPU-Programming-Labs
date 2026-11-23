NVCC        ?= nvcc
ARCH        ?=

NVCCFLAGS   = -O3 -std=c++14 $(if $(ARCH),-arch=$(ARCH)) -Xcompiler -fopenmp

all: matmul

matmul: matmul.cu ../common/cuda_check.h
	$(NVCC) $(NVCCFLAGS) matmul.cu -o $@ -lcublas

clean:
	rm -f matmul *.o
.PHONY: all clean
