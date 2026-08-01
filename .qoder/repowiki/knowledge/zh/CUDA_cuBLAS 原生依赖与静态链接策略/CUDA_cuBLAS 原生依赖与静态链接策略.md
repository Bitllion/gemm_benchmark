---
kind: dependency_management
name: CUDA/cuBLAS 原生依赖与静态链接策略
category: dependency_management
scope:
    - '**'
source_files:
    - Makefile
    - gemm_benchmark.cu
---

本仓库为单文件 CUDA 基准测试程序，未使用任何包管理器（如 pip、conda、npm、go mod、cargo 等），也不存在 lockfile、vendor 目录或私有注册表配置。依赖管理完全通过 Makefile 中的编译/链接参数声明，属于**原生系统级依赖**模式。

### 1. 使用的系统与工具链
- **编译器**: `nvcc`（CUDA Toolkit 的 NVIDIA 编译器），路径由 `CUDA_PATH` 变量控制，默认 `/usr/local/cuda`
- **架构目标**: `-arch=sm_80`，固定针对 A100 GPU
- **优化选项**: `-O3 -std=c++17 --use_fast_math`
- **运行时库**: 仅依赖 NVIDIA CUDA 运行时和 cuBLAS 库

### 2. 核心依赖声明位置
- **Makefile** 中通过 `LDFLAGS = -lcublas -Xlinker -rpath,$(CUDA_PATH)/targets/x86_64-linux/lib` 声明动态链接 cuBLAS，并设置运行时库搜索路径
- **gemm_benchmark.cu** 中通过 `#include <cublas_v2.h>`、`<cuda_fp16.h>`、`<cuda_bf16.h>`、`<cuda_fp8.h>` 引入 CUDA 头文件

### 3. 依赖版本约束
- 无显式版本号声明，依赖版本由宿主机安装的 CUDA Toolkit 决定
- 代码使用 `CUDART_VERSION` 宏在运行时打印 CUDA 版本信息，便于诊断环境差异
- 所有精度类型（FP64/FP32/TF32/FP16/BF16/INT8/FP8）均依赖对应 CUDA 版本的头文件支持

### 4. 构建与运行约定
- 编译: `make` 或 `nvcc -O3 -arch=sm_80 -lcublas gemm_benchmark.cu -o gemm_benchmark`
- 运行: `./gemm_benchmark [矩阵大小] [迭代次数]`
- 提供快捷目标: `make run`（完整测试）、`make quick`（快速测试）、`make large`（大规模测试）
- 清理: `make clean` 删除生成的可执行文件

### 5. 约束与限制
- 必须安装与 sm_80 兼容的 CUDA Toolkit（A100 要求）
- 运行时需确保 `libcublas.so` 可通过 rpath 找到，或通过 `LD_LIBRARY_PATH` 指定
- FP8 在 A100 上非原生支持，代码通过 FP8 存储 + FP16 计算模拟实现，结果仅供参考
- 无跨平台或交叉编译支持，仅限 x86_64 Linux + NVIDIA GPU 环境