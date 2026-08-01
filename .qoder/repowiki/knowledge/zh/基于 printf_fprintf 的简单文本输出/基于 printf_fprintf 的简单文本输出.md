---
kind: logging_system
name: 基于 printf/fprintf 的简单文本输出
category: logging_system
scope:
    - '**'
source_files:
    - gemm_benchmark.cu
---

该仓库未实现专门的日志系统。所有输出均通过标准 C/C++ 的 `printf` 和 `fprintf(stderr, ...)` 直接完成，没有使用任何第三方日志框架（如 spdlog、glog、nlohmann::json 等），也没有结构化日志字段、日志级别管理或日志路由机制。

具体模式：
- 错误信息：通过 `CUDA_CHECK` 和 `CUBLAS_CHECK` 宏统一输出到 `stderr`，格式为 `"[文件:行号] 调用名/状态码"`，并直接 `exit(EXIT_FAILURE)` 终止程序。
- 基准结果与提示信息：全部使用 `printf` 输出到 `stdout`，以固定宽度格式化表格形式展示 GPU 信息、各精度 GEMM 性能（GFLOPS/TFLOPS、耗时、迭代次数）及说明注释。
- 无日志级别概念：错误走 stderr，其余信息走 stdout，不存在 debug/info/warn/error 分级。
- 无结构化字段：输出为纯文本表格，不包含 JSON/XML 等可解析结构。
- 无日志开关或配置：无法在运行时启用/禁用某类输出，也无法重定向到文件或远程收集器。

约束来源：整个项目仅由单个 `gemm_benchmark.cu` 源文件和顶层 `Makefile` 构成，代码中未发现任何日志库头文件引用或初始化逻辑。