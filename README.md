# NVIDIA A100 GEMM 算力基准测试

基于 GEMM（通用矩阵乘法）测量 NVIDIA A100 GPU 在不同数据精度下的计算能力。

## 测试精度

| 精度 | 数据类型 | 说明 |
|------|---------|------|
| FP64 | 双精度浮点 | CUDA Core / Tensor Core |
| FP32 | 单精度浮点 | CUDA Core |
| TF32 | Tensor Float 32 | A100 Tensor Core 加速 FP32 计算 |
| FP16 | 半精度浮点 | Tensor Core，FP32 累加 |
| BF16 | Brain Float 16 | Tensor Core，FP32 累加 |
| INT8 | 8 位整数 | Tensor Core，INT32 累加 |
| FP8\* | 8 位浮点（模拟） | FP8 存储 + FP16 计算，A100 不原生支持 |

> **关于 FP8：** A100（sm_80，Ampere 架构）不原生支持 FP8 计算。原生 FP8 需要 Hopper（H100）或 Ada Lovelace（L40/RTX 4090）架构。本项目通过自定义 CUDA kernel 以 FP8 格式存储、FP16 计算的方式进行模拟，结果仅供参考。

## 环境要求

- **GPU：** NVIDIA A100（Compute Capability ≥ 8.0）
- **系统：** Ubuntu 22.04（或其他 Linux 发行版）
- **CUDA：** 12.0 及以上（本项目使用 CUDA 12.4）
- **NVIDIA 驱动：** 550+ 或兼容版本
- **依赖库：** cuBLAS（随 CUDA Toolkit 安装）

### 验证环境

```bash
nvcc --version          # 确认 CUDA 版本
nvidia-smi              # 确认 GPU 和驱动
```

## 编译

```bash
make
```

编译参数说明：

- 架构目标：`-arch=sm_80`（A100 Ampere）
- 优化级别：`-O3 --use_fast_math`
- 链接库：`-lcublas`

如需适配其他 GPU，修改 Makefile 中的 `ARCH` 变量：

```makefile
# A100 / A30 (Ampere)
ARCH = -arch=sm_80

# H100 (Hopper)
ARCH = -arch=sm_90

# RTX 4090 / L40 (Ada Lovelace)
ARCH = -arch=sm_89
```

## 运行

```bash
# 完整测试（矩阵 2048 / 4096 / 8192，每种 20 次迭代）
make run

# 快速测试（仅 2048，10 次迭代）
make quick

# 大矩阵测试（16384，10 次迭代）
make large

# 自定义参数
./gemm_benchmark <矩阵大小> <迭代次数>
./gemm_benchmark 8192 30        # 8192×8192 矩阵，30 次迭代
./gemm_benchmark 4096 50        # 4096×4096 矩阵，50 次迭代
```

## 输出示例

```
╔═══════════════════════════════════════════════════════════════════════╗
║           NVIDIA A100 GEMM 算力基准测试 (Benchmark)                  ║
╚═══════════════════════════════════════════════════════════════════════╝

  GPU:              NVIDIA A100-SXM4-40GB
  Compute Cap:      8.0
  SM Count:         108
  Memory:           39.4 GB
  CUDA Version:     12.4

  ┌───────────────────────────────────────────────────────────────────────┐
  │  矩阵尺寸:  8192 x  8192   元素数: 67108864   FLOPs/次: 1.10e+12    │
  └───────────────────────────────────────────────────────────────────────┘
  Type      Size            |  Performance                             |  Time
  ------------------------------------------------------------------------------------------
  FP64       8192 x  8192  |  16484.84 GFLOPS (  16.48 TFLOPS)  |   66.70 ms
  FP32       8192 x  8192  |  18858.82 GFLOPS (  18.86 TFLOPS)  |   58.30 ms
  TF32       8192 x  8192  | 113722.16 GFLOPS ( 113.72 TFLOPS)  |    9.67 ms
  FP16       8192 x  8192  | 235640.25 GFLOPS ( 235.64 TFLOPS)  |    4.67 ms
  BF16       8192 x  8192  | 244786.05 GFLOPS ( 244.79 TFLOPS)  |    4.49 ms
  INT8       8192 x  8192  |  73598.40 GFLOPS (  73.60 TFLOPS)  |   14.94 ms
  FP8*       8192 x  8192  |   2277.96 GFLOPS (   2.28 TFLOPS)  |  482.67 ms
```

## 性能参考

A100-SXM4-40GB 官方峰值算力：

| 精度 | 峰值（Dense） | 峰值（2:4 稀疏） |
|------|-------------|----------------|
| FP64 | 9.7 TFLOPS | 19.5 TFLOPS |
| FP32 | 19.5 TFLOPS | — |
| TF32 | 156 TFLOPS | 312 TFLOPS |
| FP16 | 312 TFLOPS | 624 TFLOPS |
| BF16 | 312 TFLOPS | 624 TFLOPS |
| INT8 | 624 TOPS | 1248 TOPS |

> 本测试使用 Dense GEMM（密集矩阵乘法），不涉及结构化稀疏。NVIDIA 官方峰值中部分数据标注了 2:4 稀疏加速倍数，实际 Dense 峰值约为标称稀疏峰值的 50%。

## 项目结构

```
new-gemm/
├── gemm_benchmark.cu   # 主程序（所有精度的 GEMM 基准测试）
├── Makefile            # 构建脚本
└── README.md           # 本文档
```

## 技术实现

- **FP64 / FP32 / TF32 / FP16 / BF16 / INT8**：使用 cuBLAS `cublasGemmEx` API，自动选择最优 Tensor Core 算法
- **FP8**：自定义 CUDA kernel，Tiled GEMM + Shared Memory，FP8 E4M3 格式存储、FP16 精度计算
- **计时方式**：CUDA Event 计时（`cudaEventRecord` + `cudaEventElapsedTime`），包含暖机阶段
- **FLOPs 计算**：标准 GEMM 浮点运算量公式 `2 × M × N × K`

## 已知限制

1. **INT8 性能偏低**：`cublasGemmEx` 通用 API 对 INT8 的算法选择可能不是最优的。如需接近 INT8 峰值，建议使用 `cublasLt`（cuBLAS Lightweight）接口进行精细的算法配置
2. **FP8 为模拟值**：A100 不原生支持 FP8，自定义 kernel 结果不能代表硬件原生 FP8 性能
3. **不含稀疏加速**：测试使用 Dense 矩阵，无法利用 2:4 结构化稀疏特性

## 许可证

本项目仅供学习和基准测试使用。
