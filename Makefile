# ============================================================
#  Makefile - NVIDIA A100 GEMM Benchmark (优化版)
# ============================================================

CUDA_PATH   ?= /usr/local/cuda
NVCC         = $(CUDA_PATH)/bin/nvcc
ARCH         = -arch=sm_80
NVCC_FLAGS   = -O3 $(ARCH) -std=c++17 --use_fast_math -diag-suppress 177,1650
LDFLAGS      = -lcublasLt -lcublas -Xlinker -rpath,$(CUDA_PATH)/targets/x86_64-linux/lib

TARGET       = gemm_benchmark
SRC          = gemm_benchmark.cu

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) $< -o $@
	@echo "  ✓ 编译完成: ./$(TARGET)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "  用法:"
	@echo "    make        - 编译"
	@echo "    make run    - 编译并运行"
	@echo "    make clean  - 清理"
	@echo ""
	@echo "  运行: ./$(TARGET)"
