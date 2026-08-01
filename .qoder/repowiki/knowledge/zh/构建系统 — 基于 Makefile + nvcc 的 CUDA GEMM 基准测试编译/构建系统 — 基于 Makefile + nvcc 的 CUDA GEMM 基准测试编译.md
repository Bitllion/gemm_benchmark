---
kind: build_system
name: 构建系统 — 基于 Makefile + nvcc 的 CUDA GEMM 基准测试编译
category: build_system
scope:
    - '**'
source_files:
    - Makefile
    - gemm_benchmark.cu
---

本项目采用极简的 Makefile 驱动构建系统，使用 NVIDIA CUDA 工具链（nvcc）直接编译单个 CUDA 源文件 `gemm_benchmark.cu`，并链接 cuBLAS 库生成可执行文件 `gemm_benchmark`。未使用 CMake、Docker、CI/CD 流水线或容器化方案。

**构建工具与依赖**
- 编译器：`nvcc`（CUDA Toolkit），默认路径 `/usr/local/cuda/bin/nvcc`
- 架构目标：`-arch=sm_80`（NVIDIA A100 GPU）
- 标准库：C++17，启用 `-use_fast_math` 优化
- 运行时依赖：cuBLAS 库，通过 `-lcublas` 链接，rpath 指向 `$CUDA_PATH/targets/x86_64-linux/lib`

**Makefile 目标约定**
- `make all`：编译生成 `./gemm_benchmark`
- `make run`：运行完整基准测试（默认矩阵尺寸 2048/4096/8192）
- `make quick`：快速测试（仅 2048×2048，迭代 10 次）
- `make large`：大规模测试（含 16384×16384）
- `make clean`：清理生成的可执行文件
- `make help`：打印用法说明

**自定义方式**
- 可通过环境变量 `CUDA_PATH` 指定 CUDA 安装路径（默认 `/usr/local/cuda`）
- 程序支持命令行参数：`./gemm_benchmark <矩阵大小> <迭代次数>`

**约束与限制**
- 仅支持 x86_64 Linux 平台（rpath 硬编码为 `x86_64-linux`）
- 仅针对 A100 GPU（sm_80）编译，跨架构需手动修改 `ARCH` 变量
- 无交叉编译、无多目标并行构建、无单元测试集成
- 未提供 Docker 镜像、CI 配置或包管理脚本