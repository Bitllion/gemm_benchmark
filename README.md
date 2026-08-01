# NVIDIA GPU GEMM 算力基准测试

基于 GEMM（通用矩阵乘法）测量 NVIDIA GPU 在不同数据精度下的计算能力。

## 支持的 GPU

| GPU | 架构 | Compute Cap | 编译参数 |
|-----|------|-------------|---------|
| A100 / A30 | Ampere | 8.0 | `make ARCH=a100` |
| L40 / RTX 4090 | Ada Lovelace | 8.9 | `make ARCH=ada` |
| H100 / H200 | Hopper | 9.0 | `make ARCH=h100` |
| B200 / B300 | Blackwell | 10.0 | `make ARCH=b200` |

> H100/H200/B200/B300 支持原生 FP8 GEMM；A100 使用 FP8 存储+FP16 计算模拟。

## 快速开始

```bash
make        # 编译 (自动检测当前 GPU 架构)
make run    # 运行 (或直接 ./gemm_benchmark)
```

## 命令行参数

```
用法: ./gemm_benchmark [选项]

选项:
  -d, --device <N>          指定 GPU 设备索引 (默认: 0)
      --pci <BUS_ID>        按 PCI Bus ID 选择 GPU
      --json                以 JSON 格式输出结果
  -o, --output <FILE>       输出文件路径 (配合 --json 使用)
  -h, --help                显示帮助信息
```

### 示例

```bash
# 默认运行 (文本输出)
./gemm_benchmark

# 指定 GPU 1
./gemm_benchmark -d 1

# 按 PCI 地址选择 GPU
./gemm_benchmark --pci 0000:00:08.0

# JSON 输出到终端
./gemm_benchmark --json

# JSON 写入文件
./gemm_benchmark --json -o result.json

# 组合使用
./gemm_benchmark -d 0 --json -o benchmark_result.json
```

## 测试精度

| 精度 | 说明 | 硬件支持 |
|------|------|---------|
| FP64 | 双精度浮点，Tensor Core | 全部 |
| FP32 | 单精度浮点，CUDA Core | 全部 |
| TF32 | Tensor Float 32 | Ampere+ |
| FP16 | 半精度浮点，Tensor Core（FP32 累加） | 全部 |
| BF16 | Brain Float 16，Tensor Core（FP32 累加） | Ampere+ |
| INT8 | 8 位整数，Tensor Core（INT32 累加） | 全部 |
| FP8 | 8 位浮点（原生 / 模拟） | H100+ 原生；A100 模拟 |

## 编译

```bash
# 自动检测当前 GPU (推荐)
make

# 手动指定架构
make ARCH=a100          # A100
make ARCH=h100          # H100 / H200
make ARCH=ada           # L40 / RTX 4090
make ARCH=b200          # B200 / B300
make ARCH=all           # 编译所有架构 (用于分发)
make ARCH=sm_90         # 直接指定 sm_xx
```

## JSON 输出格式

使用 `--json` 选项输出结构化 JSON 数据，包含：

```json
{
  "gpu": {
    "name": "NVIDIA A100-SXM4-40GB",
    "compute_capability": "8.0",
    "sm_count": 108,
    "memory_gb": 39.4,
    "clock_max_mhz": 1410,
    "pci_bus_id": "0000:00:08.0",
    "cuda_version": "12.4",
    "driver_version": "12.4",
    "has_native_fp8": false,
    "has_tf32": true
  },
  "config": {
    "matrix_size": 8192,
    "iterations": 20,
    "warmup": 10
  },
  "results": [
    {
      "precision": "FP64",
      "matrix_size": 8192,
      "gflops": 16484.0,
      "tflops": 16.48,
      "time_ms": 66.70,
      "iterations": 20,
      "status": "ok"
    }
  ],
  "peak_reference_tflops": {
    "FP64": 19.5, "FP32": 19.5, "TF32": 156.0,
    "FP16": 312.0, "BF16": 312.0, "INT8": 624.0
  }
}
```

## 技术实现

- **cuBLASLt + heuristic 算法选择** — 搜索 top-3 最优算法
- **GPU 自动调优** — nvidia-smi 锁定最高性能模式
- **功耗压力测试** — 15 秒持续满载 FP16 GEMM
- **多架构运行时检测** — 自动识别 GPU 能力，启用/禁用对应特性
- **FP8 双模式** — H100+ 使用 cuBLASLt 原生 FP8；A100 使用自定义内核模拟
- **设备选择** — 支持按索引或 PCI Bus ID 指定 GPU

## 环境要求

- **GPU：** NVIDIA A100 / H100 / H200 / B200 / B300 及同架构型号
- **系统：** Ubuntu 22.04+（或其他 Linux）
- **CUDA：** 12.0+
- **驱动：** 550+
- **依赖：** cuBLAS（随 CUDA Toolkit）

## 项目结构

```
new-gemm/
├── gemm_benchmark.cu   # 主程序
├── Makefile            # 构建脚本 (多架构支持)
├── README.md           # 本文档
└── result.json         # JSON 输出示例 (运行后生成)
```
