# GPU-Computing Makefile
#
#   make                 build everything (CUDA + CPU)
#   make cuda            build only CUDA binaries
#   make cpu             build only CPU baseline binaries
#   make bench           build + run the Python benchmark harness
#   make clean           remove bin/
#
# Override the arch or compiler on the command line, e.g.:
#   make ARCH=80
#   make NVCC=/usr/local/cuda-12.4/bin/nvcc

ARCH   ?= 89
NVCC   ?= nvcc
CXX    ?= g++

NVCC_FLAGS := -O3 -arch=sm_$(ARCH) -Xcompiler "-Wno-unused-result"
CXX_FLAGS  := -O3 -std=c++17 -Wall -Wno-unused-result

CUDA_SRC := $(wildcard src/cuda/*.cu)
CPU_SRC  := $(wildcard src/cpu/*.cpp)

CUDA_BIN := $(patsubst src/cuda/%.cu,bin/cuda/%,$(CUDA_SRC))
CPU_BIN  := $(patsubst src/cpu/%.cpp,bin/cpu/%,$(CPU_SRC))

.PHONY: all cuda cpu bench clean
.DEFAULT_GOAL := all

all: cuda cpu

cuda: $(CUDA_BIN)

cpu: $(CPU_BIN)

bin/cuda/%: src/cuda/%.cu | bin/cuda
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

bin/cpu/%: src/cpu/%.cpp | bin/cpu
	$(CXX) $(CXX_FLAGS) -o $@ $<

bin/cuda bin/cpu:
	@mkdir -p $@

PYTHON ?= python3

bench: all
	$(PYTHON) scripts/bench_kernels.py --config configs/bench_shapes.yaml

clean:
	rm -rf bin
