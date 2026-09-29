# make [ARCH=sm_XX]   -- ARCH берём с реального GPU кластера (nvidia-smi --query-gpu=compute_cap)
NVCC     ?= nvcc
ARCH     ?=
NVCCFLAGS = -O3 -std=c++14 $(if $(ARCH),-arch=$(ARCH)) -Xcompiler -fopenmp

all: sin_precision sin_precision_fastmath

sin_precision: sin_precision.cu ../common/cuda_check.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lgomp

# то же, но с -use_fast_math: sinf() превращается в __sinf()
sin_precision_fastmath: sin_precision.cu ../common/cuda_check.h
	$(NVCC) $(NVCCFLAGS) -use_fast_math $< -o $@ -lgomp

clean:
	rm -f sin_precision sin_precision_fastmath
.PHONY: all clean
