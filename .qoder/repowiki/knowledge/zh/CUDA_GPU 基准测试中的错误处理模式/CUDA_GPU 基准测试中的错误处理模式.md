---
kind: error_handling
name: CUDA/GPU 基准测试中的错误处理模式
category: error_handling
scope:
    - '**'
source_files:
    - gemm_benchmark.cu
---

该仓库是一个基于 cuBLAS 和自定义 CUDA 内核的 NVIDIA A100 GPU GEMM 基准测试程序，错误处理采用简洁直接的宏检查模式，而非复杂的异常或错误类型体系。

**错误处理系统**
- 使用两个核心宏进行错误检查：`CUDA_CHECK` 和 `CUBLAS_CHECK`
- `CUDA_CHECK` 包装所有 CUDA Runtime API 调用，检查返回的 `cudaError_t`，失败时打印文件、行号、调用名和错误字符串后直接 `exit(EXIT_FAILURE)`
- `CUBLAS_CHECK` 包装所有 cuBLAS API 调用，检查 `cublasStatus_t`，失败时打印状态码后同样直接退出
- 在 FP8 GEMM 路径中，对 cuBLAS 不支持的配置采用条件判断并打印提示信息后 `return` 继续执行

**关键设计决策**
- 零异常策略：不使用 C++ 异常机制，所有错误通过宏立即终止程序
- 统一错误输出格式：包含源文件名、行号、调用名称和错误信息
- 资源清理：由于程序直接退出，依赖操作系统回收资源，没有显式的资源释放错误处理
- 非致命错误处理：在 `run_cublas_gemm` 中对 cuBLAS 不支持的配置（如某些精度组合）采用条件分支跳过而非报错

**约束与约定**
- 所有 CUDA/cuBLAS 调用必须通过对应宏包装，禁止直接使用裸 API 调用
- 错误处理遵循"快速失败"原则，发现错误立即终止而非尝试恢复
- 对于硬件能力限制（如 A100 不原生支持 FP8），采用模拟实现并标注说明
- 内存分配失败等严重错误通过 `CUDA_CHECK` 捕获并终止程序