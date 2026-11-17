NVCC        ?= nvcc
ARCH        ?=
PNG_CFLAGS  ?= $(shell pkg-config --cflags libpng 2>/dev/null)
PNG_LIBS    ?= $(shell pkg-config --libs libpng 2>/dev/null || echo -lpng)

NVCCFLAGS = -O3 -std=c++14 --fmad=false $(if $(ARCH),-arch=$(ARCH)) $(PNG_CFLAGS) \
            -Xcompiler -fopenmp,-ffp-contract=off

SRC = filters_2gpu.cu ../lab2/image_io.cpp ../lab2/cpu_filters.cpp
HDR = ../lab2/filter_common.h ../lab2/image_io.h ../lab2/cpu_filters.h ../common/cuda_check.h

filters_2gpu: $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) -I../lab2 $(SRC) -o $@ $(PNG_LIBS) -lgomp

clean:
	rm -f filters_2gpu
.PHONY: clean
