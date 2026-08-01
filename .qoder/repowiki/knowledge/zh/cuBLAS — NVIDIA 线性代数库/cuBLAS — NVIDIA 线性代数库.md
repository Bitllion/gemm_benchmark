---
kind: external_dependency
name: cuBLAS — NVIDIA 线性代数库
slug: cublas
category: external_dependency
category_hints:
    - sdk_real_api
    - framework_behavior
scope:
    - '**'
---

项目通过 `cublasGemmEx` API 统一调用 FP64/FP32/TF32/FP16/BF16/INT8 的 GEMM 计算，由 cuBLAS 自动选择最优 Tensor Core 算法。已知限制：通用 API 对 INT8 的算法选择可能非最优，如需接近峰值性能应改用 `cublasLt`（cuBLAS Lightweight）接口进行精细配置。TF32 通过 `CUBLAS_COMPUTE_32F_FAST_TF32` 启用 A100 Tensor Core 加速。