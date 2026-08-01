---
kind: external_dependency
name: NVIDIA CUDA Toolkit — GPU 编译与运行时依赖
slug: nvidia-cuda-toolkit
category: external_dependency
category_hints:
    - vendor_identity
    - sdk_real_api
scope:
    - '**'
---

本项目通过 nvcc（CUDA 编译器）构建，依赖 CUDA 12.4 工具链。编译时指定 `-arch=sm_80` 目标 A100 Ampere 架构，链接 cuBLAS 库。运行时通过 `cudaEventRecord/ElapsedTime` 计时，使用 cuBLAS `cublasGemmEx` API 执行 GEMM 计算。FP8 路径使用 `__nv_fp8_e4m3` 类型及自定义 kernel 模拟（A100 不原生支持 FP8）。需确保系统安装匹配的 NVIDIA 驱动（550+）和 CUDA 12.x。