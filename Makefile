# ============================================================
#  Makefile - NVIDIA A100 GEMM Benchmark (优化版)
# ============================================================

CUDA_PATH   ?= /usr/local/cuda
NVCC         = $(CUDA_PATH)/bin/nvcc
ARCH         = -arch=sm_80
NVCC_FLAGS   = -O3 $(ARCH) -std=c++17 --use_fast_math
LDFLAGS      = -lcublasLt -lcublas -Xlinker -rpath,$(CUDA_PATH)/targets/x86_64-linux/lib

TARGET       = gemm_benchmark
SRC          = gemm_benchmark.cu

.PHONY: all clean run quick large monitor help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) $< -o $@
	@echo "  ✓ 编译完成: ./$(TARGET)"

run: $(TARGET)
	@echo "  ▶ 运行完整测试 (4096/8192/16384, 4流并发)..."
	@echo ""
	./$(TARGET)

# 快速测试 (仅 4096)
quick: $(TARGET)
	@echo "  ▶ 快速测试 (矩阵 4096, 4流)..."
	@echo ""
	./$(TARGET) 4096 10 4

# 大规模测试
large: $(TARGET)
	@echo "  ▶ 大规模测试 (矩阵 16384, 4流)..."
	@echo ""
	./$(TARGET) 16384 20 4

# 带功耗监控运行
monitor: $(TARGET)
	@echo "  ▶ 运行测试并在另一个终端监控功耗..."
	@echo "  提示: 在另一个终端运行 watch -n 0.5 nvidia-smi"
	@echo ""
	./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "  用法:"
	@echo "    make            - 编译"
	@echo "    make run        - 完整测试 (4096/8192/16384, 4流)"
	@echo "    make quick      - 快速测试 (4096, 10次迭代)"
	@echo "    make large      - 大矩阵 (16384)"
	@echo "    make monitor    - 带功耗监控提示"
	@echo "    make clean      - 清理"
	@echo ""
	@echo "  自定义: ./$(TARGET) <矩阵大小> <迭代次数> <流数量>"
	@echo "  示例:   ./$(TARGET) 8192 30 8   # 8192矩阵, 30次, 8流"
