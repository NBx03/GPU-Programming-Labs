# make [ARCH=sm_XX]   -- ARCH берём с реального GPU кластера (nvidia-smi --query-gpu=compute_cap)
# Нужны: nvcc, libpng (заголовки png.h и -lpng), OpenMP (gcc).
NVCC        ?= nvcc
ARCH        ?=
PNG_CFLAGS  ?= $(shell pkg-config --cflags libpng 2>/dev/null)
PNG_LIBS    ?= $(shell pkg-config --libs libpng 2>/dev/null || echo -lpng)

# --fmad=false и -ffp-contract=off: одинаковая float-арифметика на CPU и GPU (нет FMA-слияния),
# поэтому результат GPU должен побитово совпадать с CPU-эталоном.
NVCCFLAGS = -O3 -std=c++14 --fmad=false $(if $(ARCH),-arch=$(ARCH)) $(PNG_CFLAGS) \
            -Xcompiler -fopenmp,-ffp-contract=off

SRC = filters.cu image_io.cpp cpu_filters.cpp
HDR = filter_common.h image_io.h cpu_filters.h ../common/cuda_check.h

filters: $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(PNG_LIBS) -lgomp

clean:
	rm -f filters bench_tmp.png
.PHONY: clean
