# NVIDIA A100 GEMM 算力基准测试

基于 GEMM（通用矩阵乘法）测量 NVIDIA A100 GPU 在不同数据精度下的计算能力。

## 快速开始

```bash
make        # 编译
make run    # 运行（或直接 ./gemm_benchmark）
```

无需任何参数，程序自动选择最优配置（8192×8192 矩阵，20 次迭代）并输出结果。

## 测试精度

| 精度 | 说明 | A100 Dense 峰值 |
|------|------|----------------|
| FP64 | 双精度浮点，Tensor Core | 19.5 TFLOPS |
| FP32 | 单精度浮点，CUDA Core | 19.5 TFLOPS |
| TF32 | Tensor Float 32，A100 Tensor Core 加速 | 156 TFLOPS |
| FP16 | 半精度浮点，Tensor Core（FP32 累加） | 312 TFLOPS |
| BF16 | Brain Float 16，Tensor Core（FP32 累加） | 312 TFLOPS |
| INT8 | 8 位整数，Tensor Core（INT32 累加） | 624 TOPS |
| FP8\* | 8 位浮点（模拟），A100 不原生支持 FP8 | — |

> **关于 FP8：** A100（sm_80，Ampere 架构）不原生支持 FP8。原生 FP8 需要 Hopper（H100）或 Ada Lovelace（L40/RTX 4090）。本项目通过自定义 CUDA kernel 以 FP8 存储 + FP16 计算模拟，结果仅供参考。

## 输出示例

```
╔══════════════════════════════════════════════════════════════════════════╗
║  NVIDIA A100 GEMM 算力基准测试 v3 — cuBLASLt + 功耗优化                ║
╚══════════════════════════════════════════════════════════════════════════╝

  GPU:              NVIDIA A100-SXM4-40GB
  Compute Cap:      8.0
  SM Count:         108
  Memory:           39.4 GB
  CUDA Version:     12.4

  ┌───────────────────────────────────────────────────────────────────────┐
  │  矩阵:  8192 x  8192    FLOPs/GEMM: 1.100e+12                       │
  └───────────────────────────────────────────────────────────────────────┘
  Type      Size            |  Performance                             |  Time
  ────────────────────────────────────────────────────────────────────────────────
  FP64       8192 x  8192  |    16485 GFLOPS  (  16.49 TFLOPS)  |    66.70 ms
  FP32       8192 x  8192  |    19184 GFLOPS  (  19.18 TFLOPS)  |    57.31 ms
  TF32       8192 x  8192  |   117737 GFLOPS  ( 117.74 TFLOPS)  |     9.34 ms
  FP16       8192 x  8192  |   270215 GFLOPS  ( 270.22 TFLOPS)  |     4.07 ms
  BF16       8192 x  8192  |   247437 GFLOPS  ( 247.44 TFLOPS)  |     4.44 ms
  INT8       8192 x  8192  |    73628 GFLOPS  (  73.63 TFLOPS)  |    14.93 ms
  FP8*       8192 x  8192  |     2277 GFLOPS  (   2.28 TFLOPS)  |   482.68 ms
```

## 环境要求

- **GPU：** NVIDIA A100（Compute Capability ≥ 8.0）
- **系统：** Ubuntu 22.04（或其他 Linux）
- **CUDA：** 12.0+
- **驱动：** 550+
- **依赖：** cuBLAS（随 CUDA Toolkit）

## 技术实现

- **cuBLASLt + heuristic 算法选择**：搜索 top-3 最优算法，确保 kernel 最优
- **GPU 自动调优**：通过 nvidia-smi 自动锁定最高性能模式（P0、最高时钟、400W TDP）
- **功耗压力测试**：15 秒持续满载 FP16 GEMM，测量真实功耗
- **预分配显存**：消除运行时分配开销
- **FLOPs 计算**：标准公式 `2 × M × N × K`

## 项目结构

```
new-gemm/
├── gemm_benchmark.cu   # 主程序
├── Makefile            # 构建脚本
└── README.md           # 本文档
```

## 适配其他 GPU

修改 Makefile 中的架构参数：

```makefile
ARCH = -arch=sm_80   # A100 / A30 (Ampere)
ARCH = -arch=sm_90   # H100 (Hopper)
ARCH = -arch=sm_89   # RTX 4090 / L40 (Ada Lovelace)
```

## 已知限制

1. **INT8 性能偏低** — cuBLASLt heuristic 对 INT8 的算法选择可能非最优
2. **FP8 为模拟值** — A100 不原生支持 FP8
3. **功耗约 68% TDP** — Dense GEMM 难以完全饱和 A100 所有 SM，这是硬件特性限制
