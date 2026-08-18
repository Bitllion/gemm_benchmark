# ============================================================
#  Makefile - NVIDIA GPU GEMM Benchmark (多架构版)
# ============================================================

CUDA_PATH   ?= /usr/local/cuda
UNAME_M     := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
  CUDA_TGT   = sbsa-linux
else
  CUDA_TGT   = x86_64-linux
endif
NVCC         = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS   = -O3 -std=c++17 --use_fast_math -diag-suppress 177,1650
LDFLAGS      = -lcublasLt -lcublas -lpthread -Xlinker -rpath,$(CUDA_PATH)/targets/$(CUDA_TGT)/lib

TARGET       = gemm_benchmark
SRC          = gemm_benchmark.cu

# ============================================================
#  架构选择 (通过 make ARCH=xxx 指定)
# ============================================================
# 自动检测当前 GPU 架构
AUTO_ARCH    = $(shell $(NVCC) -arch=sm_80 --dryrun 2>&1 | grep -o 'sm_[0-9]*' | head -1)

# 默认: 当前系统 GPU
ARCH        ?= auto

# 映射: auto -> 自动检测, 或手动指定
ifeq ($(ARCH),auto)
  # 通过 nvidia-smi 检测
  GPU_CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  ifeq ($(GPU_CC),)
    GPU_ARCH = -arch=sm_80
  else
    GPU_ARCH = -arch=sm_$(subst .,,$(GPU_CC))
  endif
else ifeq ($(ARCH),a100)
  GPU_ARCH = -arch=sm_80
else ifeq ($(ARCH),h100)
  GPU_ARCH = -arch=sm_90
else ifeq ($(ARCH),h200)
  GPU_ARCH = -arch=sm_90
else ifeq ($(ARCH),ada)
  GPU_ARCH = -arch=sm_89
else ifeq ($(ARCH),b200)
  GPU_ARCH = -arch=sm_100
else ifeq ($(ARCH),b300)
  GPU_ARCH = -arch=sm_100
else ifeq ($(ARCH),all)
  # 编译所有主流架构 (用于分发)
  GPU_ARCH = -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90 \
             -gencode arch=compute_100,code=sm_100
else
  # 直接传递 sm_xx 格式
  GPU_ARCH = -arch=$(ARCH)
endif

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(LDFLAGS) $< -o $@
	@echo "  ✓ 编译完成: ./$(TARGET)  [$(GPU_ARCH)]"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "  编译:"
	@echo "    make                    # 自动检测当前 GPU 架构"
	@echo "    make ARCH=a100          # A100 (sm_80)"
	@echo "    make ARCH=h100          # H100/H200 (sm_90)"
	@echo "    make ARCH=ada           # L40/RTX4090 (sm_89)"
	@echo "    make ARCH=b200          # B200/B300 (sm_100)"
	@echo "    make ARCH=all           # 所有架构 (分发用)"
	@echo "    make ARCH=sm_80         # 手动指定"
	@echo ""
	@echo "  运行:"
	@echo "    ./$(TARGET)                           # 默认测试所有 GPU"
	@echo "    ./$(TARGET) -d 1                      # 指定 GPU 1"
	@echo "    ./$(TARGET) --pci 0000:00:08.0        # 按 PCI 地址"
	@echo "    ./$(TARGET) --json -o result.json     # JSON 输出"
	@echo "    ./$(TARGET) --help                    # 帮助"
