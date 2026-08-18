/**
 * gemm_benchmark.cu  —  NVIDIA GPU GEMM 算力基准测试
 *
 * 支持 GPU: A100 / H100 / H200 / B200 / B300 及同架构型号
 * 精度: FP64 / FP32 / TF32 / FP16 / BF16 / INT8 / FP8(H100+原生)
 *
 * 用法:
 *   ./gemm_benchmark                          # 默认: 所有 GPU, 文本输出
 *   ./gemm_benchmark --device 1               # 指定单个 GPU
 *   ./gemm_benchmark --pci 0000:00:08.0       # 按 PCI Bus ID 选择
 *   ./gemm_benchmark --json                   # 终端输出 JSON
 *   ./gemm_benchmark --json --output r.json   # JSON 写入文件
 *   ./gemm_benchmark -d 0 --json -o out.json  # 组合使用
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/* ================================================================
 *  错误检查宏
 * ================================================================ */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error [%s:%d] %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _st = (call);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS Error [%s:%d] status=%d\n", __FILE__,      \
                    __LINE__, (int)_st);                                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ================================================================
 *  常量
 * ================================================================ */
static constexpr size_t WORKSPACE_SIZE = 128ULL * 1024 * 1024;

/* ================================================================
 *  配置 & 数据结构
 * ================================================================ */
struct Config {
    int device = 0;
    bool json_output = false;
    std::string output_file;
    bool device_set = false;
    std::string pci_bus_id;
    bool pci_set = false;
};

struct ArchInfo {
    int major = 0, minor = 0;
    std::string name;
    bool has_fp8 = false;
    bool has_tf32 = false;
    double fp64_peak = 0, fp32_peak = 0, tf32_peak = 0;
    double fp16_peak = 0, bf16_peak = 0, int8_peak = 0;
};

struct BenchResult {
    std::string type;
    int N;
    double gflops, tflops, time_ms;
    int iters;
    bool ok;
};

struct GpuResult {
    int device;
    std::string name;
    std::string pci_bus_id;
    int cc_major = 0, cc_minor = 0;
    int sm_count = 0;
    double memory_gb = 0;
    int clock_mhz = 0;
    ArchInfo arch;
    std::vector<BenchResult> results;
    double stress_tflops = 0;
    int stress_count = 0;
    double stress_seconds = 0;
};

/* ================================================================
 *  命令行解析
 * ================================================================ */
static void print_usage(const char *prog) {
    printf("用法: %s [选项]\n\n", prog);
    printf("选项:\n");
    printf("  -d, --device <N>          指定 GPU 设备索引 (默认: 所有 GPU)\n");
    printf("      --pci <BUS_ID>        按 PCI Bus ID 选择 GPU\n");
    printf("      --json                以 JSON 格式输出结果\n");
    printf("  -o, --output <FILE>       输出文件路径 (配合 --json 使用)\n");
    printf("  -h, --help                显示帮助信息\n");
    printf("\n示例:\n");
    printf("  %s                           # 默认运行\n", prog);
    printf("  %s -d 1                      # 使用 GPU 1\n", prog);
    printf("  %s --json -o result.json     # JSON 写入文件\n", prog);
    printf("  %s --pci 0000:00:08.0        # 按 PCI 地址选择\n", prog);
}

static Config parse_args(int argc, char *argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            cfg.device = atoi(argv[++i]);
            cfg.device_set = true;
        } else if (arg == "--pci" && i + 1 < argc) {
            cfg.pci_bus_id = argv[++i];
            cfg.pci_set = true;
        } else if (arg == "--json") {
            cfg.json_output = true;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output_file = argv[++i];
            cfg.json_output = true;
        } else {
            fprintf(stderr, "未知参数: %s\n", arg.c_str());
            print_usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

/* ================================================================
 *  设备选择
 * ================================================================ */
static int select_device(const Config &cfg) {
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));

    if (cfg.pci_set) {
        for (int i = 0; i < dev_count; i++) {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, i));
            char bus[32];
            snprintf(bus, sizeof(bus), "%04x:%02x:%02x.0",
                     p.pciDomainID, p.pciBusID, p.pciDeviceID);
            if (cfg.pci_bus_id == bus) {
                CUDA_CHECK(cudaSetDevice(i));
                return i;
            }
        }
        fprintf(stderr, "未找到 PCI Bus ID: %s\n", cfg.pci_bus_id.c_str());
        exit(1);
    }

    int dev = cfg.device;
    if (dev < 0 || dev >= dev_count) {
        fprintf(stderr, "设备 %d 无效 (共 %d 个 GPU)\n", dev, dev_count);
        exit(1);
    }
    CUDA_CHECK(cudaSetDevice(dev));
    return dev;
}

/* ================================================================
 *  架构检测
 * ================================================================ */
static ArchInfo detect_arch(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    ArchInfo info;
    info.major = prop.major;
    info.minor = prop.minor;
    info.name = prop.name;

    // sm_80 = A100, sm_89 = Ada (L40/RTX4090), sm_90 = H100/H200,
    // sm_100 = B200/B300 (Blackwell)
    info.has_tf32 = (info.major >= 8);
    info.has_fp8  = (info.major >= 9) || (info.major == 8 && info.minor >= 9);

    int sm = prop.multiProcessorCount;
    int clk_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    double clk = clk_khz / 1e6;  // GHz

    if (info.major == 8 && info.minor == 0) {
        // A100
        info.fp64_peak = 19.5;  info.fp32_peak = 19.5;
        info.tf32_peak = 156;   info.fp16_peak = 312;
        info.bf16_peak = 312;   info.int8_peak = 624;
    } else if (info.major == 9 && info.minor == 0) {
        // H100 SXM (估算, 按 SM 数缩放)
        double scale = sm / 132.0;  // H100 SXM = 132 SMs
        info.fp64_peak = 34.0 * scale; info.fp32_peak = 67.0 * scale;
        info.tf32_peak = 989.0 * scale; info.fp16_peak = 1979.0 * scale;
        info.bf16_peak = 1979.0 * scale; info.int8_peak = 3958.0 * scale;
    } else if (info.major == 9) {
        // H200 / 其他 Hopper 变体
        double scale = sm / 132.0;
        info.fp64_peak = 34.0 * scale; info.fp32_peak = 67.0 * scale;
        info.tf32_peak = 989.0 * scale; info.fp16_peak = 1979.0 * scale;
        info.bf16_peak = 1979.0 * scale; info.int8_peak = 3958.0 * scale;
    } else if (info.major >= 10) {
        // B200/B300 (Blackwell) — 粗略估算
        double scale = sm / 192.0;
        info.fp64_peak = 40.0 * scale; info.fp32_peak = 80.0 * scale;
        info.tf32_peak = 2250.0 * scale; info.fp16_peak = 4500.0 * scale;
        info.bf16_peak = 4500.0 * scale; info.int8_peak = 9000.0 * scale;
    } else {
        // 未知架构, 设为 0
        info.fp64_peak = 0; info.fp32_peak = 0;
        info.tf32_peak = 0; info.fp16_peak = 0;
        info.bf16_peak = 0; info.int8_peak = 0;
    }

    return info;
}

static std::string get_pci_bus_id(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    char buf[32];
    snprintf(buf, sizeof(buf), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    return buf;
}

/* ================================================================
 *  GPU 性能模式 (nvidia-smi)
 * ================================================================ */
static int run_silent(const char *cmd) { return WEXITSTATUS(system(cmd)); }

static void setup_gpu_performance(const std::string &busId, int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -pm 1 >/dev/null 2>&1", busId.c_str());
    run_silent(cmd);
    int clk_mhz = 0, mem_mhz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_mhz, cudaDevAttrClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_mhz, cudaDevAttrMemoryClockRate, dev));
    clk_mhz /= 1000;  // kHz -> MHz
    mem_mhz /= 1000;
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --lock-gpu-clocks=%d,%d >/dev/null 2>&1",
             busId.c_str(), clk_mhz, clk_mhz);
    run_silent(cmd);
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --applications-clocks=%d,%d >/dev/null 2>&1",
             busId.c_str(), mem_mhz, clk_mhz);
    run_silent(cmd);
    /* 功耗上限提到该卡最大值。原 -pl 400||700 是 A100/H100 的老值,
     * 套在 B300 (max 1400W) 上会锁死性能 (~57% 峰值)。 */
    snprintf(cmd, sizeof(cmd),
             "pl=$(nvidia-smi -i %s --query-gpu=power.max_limit --format=csv,noheader 2>/dev/null | awk '{print int($1)}'); "
             "[ -n \"$pl\" ] && nvidia-smi -i %s -pl \"$pl\" >/dev/null 2>&1; true",
             busId.c_str(), busId.c_str());
    run_silent(cmd);
}

static void restore_gpu(const std::string &busId) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -rgc >/dev/null 2>&1", busId.c_str());
    run_silent(cmd);
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -rac >/dev/null 2>&1", busId.c_str());
    run_silent(cmd);
    /* 功耗上限恢复默认值 (基准会把上限提到 max, 测完还原, 避免卡在峰值限流) */
    snprintf(cmd, sizeof(cmd),
             "pl=$(nvidia-smi -i %s --query-gpu=power.default_limit --format=csv,noheader 2>/dev/null | awk '{print int($1)}'); "
             "[ -n \"$pl\" ] && nvidia-smi -i %s -pl \"$pl\" >/dev/null 2>&1; true",
             busId.c_str(), busId.c_str());
    run_silent(cmd);
}

/* ================================================================
 *  工具函数
 * ================================================================ */
static inline double gemm_flops(int M, int N, int K) {
    return 2.0 * M * N * K;
}

static void init_double(double *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

/* 所有框线统一为 80 展示宽度 ("  +" + 76个"-" + "+") */
#define BOX76 "  +----------------------------------------------------------------------------+"

static void print_result(const char *label, int N, double gflops, double ms, int iters) {
    /* 整行宽度 80: "  |  TYPE  NNNNNxNNNNN  | GFLOPS (TFLOPS) | ms  i  |" */
    printf("  |  %-4s  %5dx%5d  | %10.1f GFLOPS (%7.2f TFLOPS) | %7.2fms %3di  |\n",
           label, N, N, gflops, gflops / 1000.0, ms, iters);
}

/* ================================================================
 *  cuBLASLt GEMM (heuristic 算法选择 + 候选微调)
 * ================================================================ */

/* heuristic 的排名不一定最优 (INT8 实测第 2 个候选快 ~2x), 首次遇到某配置时
 * 对各候选做 2 次迭代计时选最快, 之后同配置复用 (结果按配置确定性返回) */
struct TunedAlgo {
    unsigned long long key = 0;
    cublasLtMatmulAlgo_t algo{};
    bool valid = false;
};
static TunedAlgo g_tuned_algo;

static unsigned long long algo_key(int dev,
                                   cudaDataType_t a, cudaDataType_t b, cudaDataType_t c,
                                   cublasComputeType_t ct, int N)
{
    return ((unsigned long long)dev << 56) | ((unsigned long long)a << 44) |
           ((unsigned long long)b << 32) | ((unsigned long long)c << 20) |
           ((unsigned long long)ct << 8) | (unsigned)N;
}

static cublasStatus_t run_cublaslt_gemm(
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace, size_t ws_size,
    int N,
    cudaDataType_t a_type, cudaDataType_t b_type,
    cudaDataType_t c_type, cublasComputeType_t compute_type,
    const void *alpha, const void *beta,
    const void *d_A, const void *d_B, void *d_C,
    const void *scaleA = nullptr, const void *scaleB = nullptr)
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t results[8];
    int count = 0;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    /* desc 的 scale type 必须与 compute type 匹配 (CUDA 13 强制):
     * 64F -> CUDA_R_64F, 32I -> CUDA_R_32I (INT8 路径), 其余 -> CUDA_R_32F */
    cudaDataType_t scale_type = CUDA_R_32F;
    if (compute_type == CUBLAS_COMPUTE_64F) scale_type = CUDA_R_64F;
    else if (compute_type == CUBLAS_COMPUTE_32I) scale_type = CUDA_R_32I;

    status = cublasLtMatmulDescCreate(&opDesc, compute_type, scale_type);
    if (status != CUBLAS_STATUS_SUCCESS) goto done;

    /* FP8 输入需显式 scale 指针 (FP32 标量 1.0), 否则 heuristic 无解 */
    if (scaleA) {
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &scaleA, sizeof(scaleA));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &scaleB, sizeof(scaleB));
    }

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, a_type, N, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, b_type, N, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, c_type, N, N, N));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
        pref, 8, results, &count);

    if (status != CUBLAS_STATUS_SUCCESS || count == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        goto done;
    }

    /* 候选微调: 排名不一定最优, 首次按 (dev, 类型, N) 计时选最快 */
    {
        int dev = 0;
        cudaGetDevice(&dev);
        unsigned long long k = algo_key(dev, a_type, b_type, c_type, compute_type, N);
        if (!g_tuned_algo.valid || g_tuned_algo.key != k) {
            cudaStreamSynchronize(stream);
            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            float best_ms = 1e30f;
            int best_i = 0;
            for (int i = 0; i < count; i++) {
                cudaEventRecord(ev0, stream);
                for (int t = 0; t < 2; t++)
                    cublasLtMatmul(ltHandle, opDesc,
                                   alpha, d_A, Adesc, d_B, Bdesc,
                                   beta, d_C, Cdesc, d_C, Cdesc,
                                   &results[i].algo, workspace, ws_size, stream);
                cudaEventRecord(ev1, stream);
                cudaEventSynchronize(ev1);
                float ms = 0;
                cudaEventElapsedTime(&ms, ev0, ev1);
                if (ms < best_ms) { best_ms = ms; best_i = i; }
            }
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            g_tuned_algo.key = k;
            g_tuned_algo.algo = results[best_i].algo;
            g_tuned_algo.valid = true;
        }
    }

    status = cublasLtMatmul(ltHandle, opDesc,
                            alpha, d_A, Adesc, d_B, Bdesc,
                            beta, d_C, Cdesc, d_C, Cdesc,
                            &g_tuned_algo.algo, workspace, ws_size, stream);

done:
    if (pref)    cublasLtMatmulPreferenceDestroy(pref);
    if (Adesc)   cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc)   cublasLtMatrixLayoutDestroy(Bdesc);
    if (Cdesc)   cublasLtMatrixLayoutDestroy(Cdesc);
    if (opDesc)  cublasLtMatmulDescDestroy(opDesc);
    return status;
}

/* ================================================================
 *  单流 GEMM 基准测试
 * ================================================================ */
static BenchResult bench_gemm(
    const char *label,
    int N, int warmup, int iters,
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace,
    cudaDataType_t a_type, cudaDataType_t b_type,
    cudaDataType_t c_type, cublasComputeType_t compute_type,
    const void *alpha, const void *beta,
    void *d_A, void *d_B, void *d_C,
    bool silent = false)
{
    BenchResult r = {label, N, 0, 0, 0, iters, false};

    for (int w = 0; w < warmup; w++) {
        cublasStatus_t st = run_cublaslt_gemm(
            ltHandle, stream, workspace, WORKSPACE_SIZE,
            N, a_type, b_type, c_type, compute_type,
            alpha, beta, d_A, d_B, d_C);
        if (st != CUBLAS_STATUS_SUCCESS) {
            if (!silent)
                printf("  %-8s  %5d x %5d  |  *** 不支持此配置 (status=%d) ***\n",
                       label, N, N, (int)st);
            return r;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        run_cublaslt_gemm(ltHandle, stream, workspace, WORKSPACE_SIZE,
                          N, a_type, b_type, c_type, compute_type,
                          alpha, beta, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = total_ms / iters;
    r.gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);
    r.tflops = r.gflops / 1000.0;
    r.time_ms = avg_ms;
    r.ok = true;

    if (!silent)
        print_result(label, N, r.gflops, avg_ms, iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return r;
}

/* ================================================================
 *  FP8 GEMM — H100/H200/B200 原生 (cuBLASLt)
 * ================================================================ */
static BenchResult bench_fp8_native(
    int N, int warmup, int iters,
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace,
    void *d_A, void *d_B, void *d_C,
    bool silent = false)
{
    BenchResult r = {"FP8", N, 0, 0, 0, iters, false};
    cublasStatus_t st = CUBLAS_STATUS_NOT_SUPPORTED;

    /* CUDA_R_8F_E4M3 是枚举值不是宏, #ifdef 永远为假; 用 CUDART_VERSION 宏判断 */
#if CUDART_VERSION >= 11000
    float alpha = 1.0f, beta = 0.0f;

    /* FP8 输入必须提供 scale 指针 (FP32 标量 1.0) */
    float *d_scale = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scale, 2 * sizeof(float)));
    float h_scale[2] = {1.0f, 1.0f};
    CUDA_CHECK(cudaMemcpy(d_scale, h_scale, 2 * sizeof(float), cudaMemcpyHostToDevice));

    // FP8 E4M3 输入, FP32 输出, FP32 累加 (CUDA 13: FP8 不能配 CUBLAS_COMPUTE_16F)
    for (int w = 0; w < warmup; w++) {
        st = run_cublaslt_gemm(ltHandle, stream, workspace, WORKSPACE_SIZE,
                               N, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3,
                               CUDA_R_32F, CUBLAS_COMPUTE_32F,
                               &alpha, &beta, d_A, d_B, d_C,
                               d_scale, d_scale + 1);
        if (st != CUBLAS_STATUS_SUCCESS) break;
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iters; i++) {
            run_cublaslt_gemm(ltHandle, stream, workspace, WORKSPACE_SIZE,
                              N, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3,
                              CUDA_R_32F, CUBLAS_COMPUTE_32F,
                              &alpha, &beta, d_A, d_B, d_C,
                              d_scale, d_scale + 1);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double avg_ms = total_ms / iters;
        r.gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);
        r.tflops = r.gflops / 1000.0;
        r.time_ms = avg_ms;
        r.ok = true;
        if (!silent)
            print_result("FP8", N, r.gflops, avg_ms, iters);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_scale));
        return r;
    }
    CUDA_CHECK(cudaFree(d_scale));
#endif
    // Fallback: 不支持原生 FP8
    if (!silent)
        printf("  %-8s  %5d x %5d  |  *** cuBLASLt 原生 FP8 不可用 (status=%d) ***\n", "FP8", N, N, (int)st);
    return r;
}

/* ================================================================
 *  FP8 GEMM — 自定义内核 (A100 等不支持 FP8 的 GPU)
 * ================================================================ */
template <int TILE = 32>
__global__ void fp8_gemm_kernel(const __nv_fp8_e4m3 *__restrict__ A,
                                const __nv_fp8_e4m3 *__restrict__ B,
                                __half *__restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ac = t * TILE + threadIdx.x;
        int br = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ac < K) ? (float)A[ac * M + row] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (br < K && col < N) ? (float)B[col * K + br] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[col * M + row] = __float2half(acc);
}

__global__ void f16_to_fp8_kernel(const __half *src, __nv_fp8_e4m3 *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(src[i]);
        v = fminf(fmaxf(v * 100.0f, -448.0f), 448.0f);
        dst[i] = __nv_fp8_e4m3(v);
    }
}

static BenchResult bench_fp8_simulated(int N, int warmup, int iters, cudaStream_t stream,
                                        bool silent = false) {
    BenchResult r = {"FP8*", N, 0, 0, 0, iters, false};
    constexpr int TILE = 32;
    size_t n_elem = (size_t)N * N;
    size_t fp8_sz = n_elem * sizeof(__nv_fp8_e4m3);
    size_t f16_sz = n_elem * sizeof(__half);

    __nv_fp8_e4m3 *dA, *dB;
    __half *dC, *dA16, *dB16;
    CUDA_CHECK(cudaMalloc(&dA, fp8_sz));
    CUDA_CHECK(cudaMalloc(&dB, fp8_sz));
    CUDA_CHECK(cudaMalloc(&dC, f16_sz));
    CUDA_CHECK(cudaMalloc(&dA16, f16_sz));
    CUDA_CHECK(cudaMalloc(&dB16, f16_sz));

    __half *h = (__half *)malloc(f16_sz);
    for (size_t i = 0; i < n_elem; i++)
        h[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    CUDA_CHECK(cudaMemcpy(dA16, h, f16_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB16, h, f16_sz, cudaMemcpyHostToDevice));
    free(h);

    int t = 256, b = (n_elem + t - 1) / t;
    f16_to_fp8_kernel<<<b, t, 0, stream>>>(dA16, dA, n_elem);
    f16_to_fp8_kernel<<<b, t, 0, stream>>>(dB16, dB, n_elem);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 blk(TILE, TILE);

    for (int w = 0; w < warmup; w++)
        fp8_gemm_kernel<TILE><<<grid, blk, 0, stream>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++)
        fp8_gemm_kernel<TILE><<<grid, blk, 0, stream>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = total_ms / iters;
    r.gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);
    r.tflops = r.gflops / 1000.0;
    r.time_ms = avg_ms;
    r.ok = true;

    if (!silent)
        print_result("FP8*", N, r.gflops, avg_ms, iters);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dA16); cudaFree(dB16);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return r;
}

/* ================================================================
 *  功耗压力测试
 * ================================================================ */
static void power_stress_test(
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace,
    void *d_A, void *d_B, void *d_C, int N, double duration_sec)
{
    float alpha = 1.0f, beta = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    int count = 0;
    while (true) {
        run_cublaslt_gemm(ltHandle, stream, workspace, WORKSPACE_SIZE,
                          N, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                          CUBLAS_COMPUTE_32F, &alpha, &beta, d_A, d_B, d_C);
        count++;
        auto t1 = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(t1 - t0).count() >= duration_sec) break;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double tflops_val = gemm_flops(N, N, N) * count / (elapsed * 1e12);
    printf("  [功耗压力] %d 次 GEMM, %.1f 秒, 平均 %.2f TFLOPS\n",
           count, elapsed, tflops_val);
}

/* ================================================================
 *  功耗读取
 * ================================================================ */
static void read_power(const std::string &busId) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --query-gpu=power.draw,power.limit,clocks.sm,temperature.gpu "
             "--format=csv,noheader 2>/dev/null", busId.c_str());
    printf("  [功耗] ");
    fflush(stdout);
    run_silent(cmd);
}

static void read_power_to_buf(const std::string &busId, char *buf, size_t bufsz) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --query-gpu=power.draw,power.limit,clocks.sm,temperature.gpu "
             "--format=csv,noheader 2>/dev/null", busId.c_str());
    FILE *fp = popen(cmd, "r");
    if (!fp) { snprintf(buf, bufsz, "N/A"); return; }
    if (!fgets(buf, (int)bufsz, fp)) buf[0] = '\0';
    pclose(fp);
    /* 去掉末尾换行 */
    size_t n = strlen(buf);
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r' || buf[n-1] == ' ')) buf[--n] = '\0';
}

/* ================================================================
 *  JSON 输出 (多 GPU)
 * ================================================================ */
static std::string escape_json(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

static void write_json(const std::string &path,
                       const std::string &cuda_ver, const std::string &driver_ver,
                       int N, int iters, int warmup,
                       const std::vector<GpuResult> &gpu_results) {
    FILE *f = path.empty() ? stdout : fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "无法打开文件: %s\n", path.c_str());
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"gpu_count\": %d,\n", (int)gpu_results.size());
    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"matrix_size\": %d,\n", N);
    fprintf(f, "    \"iterations\": %d,\n", iters);
    fprintf(f, "    \"warmup\": %d,\n", warmup);
    fprintf(f, "    \"cuda_version\": \"%s\",\n", cuda_ver.c_str());
    fprintf(f, "    \"driver_version\": \"%s\"\n", driver_ver.c_str());
    fprintf(f, "  },\n");
    fprintf(f, "  \"gpus\": [\n");
    for (size_t g = 0; g < gpu_results.size(); g++) {
        const auto &gr = gpu_results[g];
        fprintf(f, "    {\n");
        fprintf(f, "      \"device_index\": %d,\n", gr.device);
        fprintf(f, "      \"name\": \"%s\",\n", escape_json(gr.name).c_str());
        fprintf(f, "      \"compute_capability\": \"%d.%d\",\n", gr.cc_major, gr.cc_minor);
        fprintf(f, "      \"sm_count\": %d,\n", gr.sm_count);
        fprintf(f, "      \"memory_gb\": %.1f,\n", gr.memory_gb);
        fprintf(f, "      \"clock_max_mhz\": %d,\n", gr.clock_mhz);
        fprintf(f, "      \"pci_bus_id\": \"%s\",\n", gr.pci_bus_id.c_str());
        fprintf(f, "      \"has_native_fp8\": %s,\n", gr.arch.has_fp8 ? "true" : "false");
        fprintf(f, "      \"has_tf32\": %s,\n", gr.arch.has_tf32 ? "true" : "false");
        fprintf(f, "      \"results\": [\n");
        for (size_t i = 0; i < gr.results.size(); i++) {
            const auto &r = gr.results[i];
            fprintf(f, "        {\n");
            fprintf(f, "          \"precision\": \"%s\",\n", r.type.c_str());
            fprintf(f, "          \"matrix_size\": %d,\n", r.N);
            fprintf(f, "          \"gflops\": %.1f,\n", r.gflops);
            fprintf(f, "          \"tflops\": %.2f,\n", r.tflops);
            fprintf(f, "          \"time_ms\": %.2f,\n", r.time_ms);
            fprintf(f, "          \"iterations\": %d,\n", r.iters);
            fprintf(f, "          \"status\": \"%s\"\n", r.ok ? "ok" : "unsupported");
            fprintf(f, "        }%s\n", (i + 1 < gr.results.size()) ? "," : "");
        }
        fprintf(f, "      ]\n");
        fprintf(f, "    }%s\n", (g + 1 < gpu_results.size()) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    if (!path.empty()) {
        fclose(f);
        printf("  [JSON] 结果已写入: %s\n", path.c_str());
    }
}

/* ================================================================
 *  单 GPU 基准测试 (封装为独立函数)
 * ================================================================ */
static GpuResult run_benchmark_on_device(int dev, bool json_output,
                                          int N, int iters, int warmup,
                                          bool silent = false) {
    GpuResult gr;
    gr.device = dev;

    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    gr.name = prop.name;
    gr.pci_bus_id = get_pci_bus_id(dev);
    gr.cc_major = prop.major;
    gr.cc_minor = prop.minor;
    gr.sm_count = prop.multiProcessorCount;
    gr.memory_gb = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    int clk_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    gr.clock_mhz = clk_khz / 1000;
    gr.arch = detect_arch(dev);

    /* ---------- 文本模式: GPU 信息头 ---------- */
    if (!json_output && !silent) {
        printf("%s\n", BOX76);
        printf("  |  GPU %d: %-67s|\n", dev, prop.name);
        printf("%s\n", BOX76);
        printf("  |  Device Index : %-59d|\n", dev);
        printf("  |  PCI Bus ID   : %-59s|\n", gr.pci_bus_id.c_str());
        char cc_buf[32];
        snprintf(cc_buf, sizeof(cc_buf), "%d.%d%s",
                 gr.cc_major, gr.cc_minor, gr.arch.has_fp8 ? " (含 FP8)" : " (无 FP8)");
        printf("  |  Compute Cap  : %-60s|\n", cc_buf);
        printf("  |  SM Count     : %-59d|\n", gr.sm_count);
        char clk_buf[32]; snprintf(clk_buf, sizeof(clk_buf), "%d MHz", gr.clock_mhz);
        printf("  |  Clock (max)  : %-59s|\n", clk_buf);
        char mem_buf[32]; snprintf(mem_buf, sizeof(mem_buf), "%.1f GB", gr.memory_gb);
        printf("  |  Memory       : %-59s|\n", mem_buf);
        /* 不闭合底线，等调优完成后继续展示矩阵信息 */
    }

    // GPU 调优始终执行 (确保性能正确)
    if (!json_output)
        setup_gpu_performance(gr.pci_bus_id, dev);

    if (!json_output && !silent)
        printf("  |  [GPU %d 调优] 完成%-57s|\n", dev, "");

    /* ---------- 初始化 cuBLASLt ---------- */
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    void *workspace;
    CUDA_CHECK(cudaMalloc(&workspace, WORKSPACE_SIZE));

    /* ---------- 预分配显存 ---------- */
    size_t max_elem = (size_t)N * N;
    void *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, max_elem * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, max_elem * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, max_elem * sizeof(double)));
    {
        double *hd = (double *)malloc(max_elem * sizeof(double));
        init_double(hd, max_elem);
        CUDA_CHECK(cudaMemcpy(d_A, hd, max_elem * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, hd, max_elem * sizeof(double), cudaMemcpyHostToDevice));
        free(hd);
    }

    /* ============================================================
     *  逐精度基准测试
     * ============================================================ */
    std::vector<BenchResult> &results = gr.results;

    if (!json_output && !silent) {
        printf("%s\n", BOX76);
        char flopbuf[32]; snprintf(flopbuf, sizeof(flopbuf), "%.3e", gemm_flops(N, N, N));
        printf("  |  Matrix: %5dx%-5d  FLOPs/GEMM: %-41s|\n", N, N, flopbuf);
        printf("%s\n", BOX76);
        printf("  | Type   Size         |   GFLOPS       (TFLOPS)            |   Time(ms)  iter|\n");
        printf("%s\n", BOX76);
    }

    // FP64
    {
        double alpha = 1.0, beta = 0.0;
        results.push_back(bench_gemm("FP64", N, warmup, iters, ltHandle, stream, workspace,
            CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F,
            &alpha, &beta, d_A, d_B, d_C, silent));
    }
    // FP32
    {
        float alpha = 1.0f, beta = 0.0f;
        results.push_back(bench_gemm("FP32", N, warmup, iters, ltHandle, stream, workspace,
            CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F,
            &alpha, &beta, d_A, d_B, d_C, silent));
    }
    // TF32
    if (gr.arch.has_tf32) {
        float alpha = 1.0f, beta = 0.0f;
        results.push_back(bench_gemm("TF32", N, warmup, iters, ltHandle, stream, workspace,
            CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
            &alpha, &beta, d_A, d_B, d_C, silent));
    }
    // FP16
    {
        float alpha = 1.0f, beta = 0.0f;
        results.push_back(bench_gemm("FP16", N, warmup, iters, ltHandle, stream, workspace,
            CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F,
            &alpha, &beta, d_A, d_B, d_C, silent));
    }
    // BF16
    {
        float alpha = 1.0f, beta = 0.0f;
        results.push_back(bench_gemm("BF16", N, warmup, iters, ltHandle, stream, workspace,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
            &alpha, &beta, d_A, d_B, d_C, silent));
    }
    // INT8 (CUBLAS_COMPUTE_32I 要求 int32 alpha/beta + scale type 32I)
    {
        int32_t alpha = 1, beta = 0;
        results.push_back(bench_gemm("INT8", N, warmup, iters, ltHandle, stream, workspace,
            CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I,
            &alpha, &beta, d_A, d_B, d_C, silent));
    }
    // FP8
    if (gr.arch.has_fp8) {
        results.push_back(bench_fp8_native(N, warmup, iters, ltHandle, stream, workspace,
                                           d_A, d_B, d_C, silent));
    } else {
        results.push_back(bench_fp8_simulated(N, warmup, iters, stream, silent));
    }

    if (!json_output && !silent) {
        printf("%s\n", BOX76);
        char pwbuf[128] = "";
        read_power_to_buf(gr.pci_bus_id, pwbuf, sizeof(pwbuf));
        printf("  |  功耗实测: %-64s|\n", pwbuf);
    }

    /* ---------- 功耗压力测试 ---------- */
    if (!json_output && !silent) {
        printf("%s\n", BOX76);
        printf("  |  压力测试: FP16 %dx%d GEMM 15 秒%-39s|\n", N, N, "");
    }
    if (!json_output) {
        // 功耗压力测试始终运行 (并行模式也测)
        float alpha_s = 1.0f, beta_s = 0.0f;
        auto t0 = std::chrono::steady_clock::now();
        int count = 0;
        while (true) {
            run_cublaslt_gemm(ltHandle, stream, workspace, WORKSPACE_SIZE,
                              N, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                              CUBLAS_COMPUTE_32F, &alpha_s, &beta_s, d_A, d_B, d_C);
            count++;
            auto t1 = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(t1 - t0).count() >= 15.0) break;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        gr.stress_tflops = gemm_flops(N, N, N) * count / (elapsed * 1e12);
        gr.stress_count = count;
        gr.stress_seconds = elapsed;
        if (!silent) {
            char resbuf[80], pw2buf[128] = "";
            snprintf(resbuf, sizeof(resbuf), "%d 次, %.1f 秒, 平均 %.2f TFLOPS",
                     count, elapsed, gr.stress_tflops);
            printf("  |  压力结果: %-68s|\n", resbuf);
            read_power_to_buf(gr.pci_bus_id, pw2buf, sizeof(pw2buf));
            printf("  |  负载功耗: %-64s|\n", pw2buf);
            printf("%s\n\n", BOX76);
        }
    }

    /* ---------- 清理 ---------- */
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(workspace);
    cudaStreamDestroy(stream);
    cublasLtDestroy(ltHandle);
    restore_gpu(gr.pci_bus_id);

    return gr;
}

/* ================================================================
 *  打印单个 GPU 的详细文本结果 (并行测试后顺序输出)
 * ================================================================ */

static void print_device_text_results(const GpuResult &gr, int N, int gpu_num, int total_gpus) {
    if (total_gpus > 1)
        printf("\n  ---  GPU %d / %d  ---\n", gpu_num, total_gpus);

    /* ---- GPU 信息框 (76 内宽) ---- */
    printf("%s\n", BOX76);
    printf("  |  GPU %d: %-68s|\n", gr.device, gr.name.c_str());
    printf("%s\n", BOX76);

    std::string cc = std::to_string(gr.cc_major) + "." + std::to_string(gr.cc_minor)
                     + (gr.arch.has_fp8 ? " (含 FP8)" : " (无 FP8)");
    char mem_buf[32]; snprintf(mem_buf, sizeof(mem_buf), "%.1f GB", gr.memory_gb);
    char clk_buf[32]; snprintf(clk_buf, sizeof(clk_buf), "%d MHz", gr.clock_mhz);

    printf("  |  Device Index : %-59d|\n", gr.device);
    printf("  |  PCI Bus ID   : %-59s|\n", gr.pci_bus_id.c_str());
    printf("  |  Compute Cap  : %-59s|\n", cc.c_str());
    printf("  |  SM Count     : %-59d|\n", gr.sm_count);
    printf("  |  Clock (max)  : %-59s|\n", clk_buf);
    printf("  |  Memory       : %-59s|\n", mem_buf);
    /* 不闭合底线，等调优完成后继续展示矩阵信息 */
    printf("%s\n", BOX76);
    printf("  |  [GPU %d 调优] 完成%-54s|\n", gr.device, "");

    /* ---- 测试矩阵信息 ---- */
    printf("%s\n", BOX76);
    char flopbuf[32]; snprintf(flopbuf, sizeof(flopbuf), "%.3e", gemm_flops(N, N, N));
    printf("  |  Matrix: %5dx%-5d  FLOPs/GEMM: %-41s|\n", N, N, flopbuf);
    printf("%s\n", BOX76);

    /* ---- 表头 ---- */
    printf("  | Type   Size         |   GFLOPS       (TFLOPS)            |   Time(ms)  iter|\n");
    printf("%s\n", BOX76);

    for (const auto &r : gr.results) {
        if (r.ok) {
            print_result(r.type.c_str(), r.N, r.gflops, r.time_ms, r.iters);
        } else {
            printf("  |  %-4s  %5dx%5d  | *** 不支持 ***%-38s|\n",
                   r.type.c_str(), r.N, r.N, "");
        }
    }

    printf("%s\n", BOX76);
    if (gr.stress_seconds > 0) {
        char resbuf[80];
        snprintf(resbuf, sizeof(resbuf), "%d 次, %.1f 秒, 平均 %.2f TFLOPS",
                 gr.stress_count, gr.stress_seconds, gr.stress_tflops);
        printf("  |  压力结果: %-61s|\n", resbuf);
        printf("%s\n", BOX76);
    }
    printf("\n");
}

/* ================================================================
 *  主函数
 * ================================================================ */
int main(int argc, char *argv[]) {
    /* ---------- 解析参数 ---------- */
    Config cfg = parse_args(argc, argv);

    /* ---------- 查询可用 GPU 数量 ---------- */
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        fprintf(stderr, "错误: 未检测到可用 GPU 设备\n");
        return EXIT_FAILURE;
    }

    /* ---------- 确定测试 GPU 列表 ---------- */
    std::vector<int> devices;
    if (cfg.pci_set || cfg.device_set) {
        int dev = select_device(cfg);
        devices.push_back(dev);
    } else {
        for (int i = 0; i < dev_count; i++) {
            devices.push_back(i);
        }
    }

    /* ---------- 驱动 & CUDA 版本 ---------- */
    /* R610 起 cudaDriverGetVersion() 返回 CUDA UMD 版本(如 13.3)而非驱动版本号;
     * 真实驱动版本从 nvidia-smi driver_version 字段取(如 610.43.02) */
    char driver_ver[32] = "unknown";
    FILE *fp = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1", "r");
    if (fp) {
        if (!fgets(driver_ver, sizeof(driver_ver), fp))
            snprintf(driver_ver, sizeof(driver_ver), "unknown");
        pclose(fp);
        size_t n = strlen(driver_ver);
        while (n > 0 && (driver_ver[n-1] == '\n' || driver_ver[n-1] == '\r' || driver_ver[n-1] == ' '))
            driver_ver[--n] = '\0';
    }
    char cuda_ver[32];
    snprintf(cuda_ver, sizeof(cuda_ver), "%d.%d",
             CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    /* ---------- 基准参数 ---------- */
    const int N = 8192;
    const int iters = 20;
    const int warmup = 10;

    /* ---------- 文本头 (非 JSON 模式) ---------- */
    if (!cfg.json_output) {
        printf("\n");
        printf("  +==========================================================================="
               "=+\n");
        printf("  |  NVIDIA GPU GEMM 算力基准测试  —  cuBLASLt + 多架构支持                    |\n");
        printf("  +==========================================================================="
               "=+\n\n");
        printf("  CUDA:             %s\n", cuda_ver);
        printf("  Driver:           %s\n", driver_ver);
        if ((int)devices.size() == dev_count) {
            printf("  检测 GPU:         %d 个 (全部顺序测试)\n", dev_count);
        } else {
            printf("  指定 GPU:         Device %d\n", devices[0]);
        }
        printf("  矩阵: %d x %d  |  迭代: %d  |  暖机: %d\n\n", N, N, iters, warmup);
    }

    /* ============================================================
     *  遍历所有目标 GPU 执行基准测试
     * ============================================================ */
    std::vector<GpuResult> all_results(devices.size());

    /* ============================================================
     *  逐 GPU 顺序执行基准测试
     * ============================================================ */
    for (size_t i = 0; i < devices.size(); i++) {
        if (!cfg.json_output && (int)devices.size() > 1) {
            printf("\n  ===============  GPU %zu / %zu  ===============\n\n",
                   i + 1, devices.size());
        }
        try {
            all_results[i] = run_benchmark_on_device(
                devices[i], cfg.json_output, N, iters, warmup, /*silent=*/false);
        } catch (...) {
            fprintf(stderr, "GPU %d 测试失败, 跳过\n", devices[i]);
            GpuResult err_gr;
            err_gr.device = devices[i];
            err_gr.name = "ERROR";
            all_results[i] = err_gr;
        }
    }

    /* ============================================================
     *  输出结果
     * ============================================================ */
    if (cfg.json_output) {
        write_json(cfg.output_file, cuda_ver, driver_ver, N, iters, warmup, all_results);
    } else {
        /* 每 GPU 摘要 */
        printf("\n");
        printf("  +============================================================================+\n");
        printf("  |  测试结果摘要                                                              |\n");
        printf("  +============================================================================+\n");
        for (size_t gi = 0; gi < all_results.size(); gi++) {
            const auto &gr = all_results[gi];
            printf("  |  GPU %d: %-67s|\n", gr.device, gr.name.c_str());
            for (const auto &r : gr.results) {
                if (r.ok) {
                    printf("  |    %-5s  %10.1f GFLOPS (%7.2f TFLOPS)%-31s|\n",
                           r.type.c_str(), r.gflops, r.tflops, "");
                }
            }
            if (gi + 1 < all_results.size())
                printf("  +============================================================================+\n");
        }
        printf("  +============================================================================+\n");
        printf("  |  说明: 所有精度均使用 cuBLASLt + heuristic 算法选择 (top-3 最优)           |\n");
        printf("  |  FP8*: 模拟 (FP8存储+FP16计算)   FP8: 原生 (H100/H200/B200)                |\n");
        printf("  +============================================================================+\n");
    }

    /* ---------- 清理所有设备 ---------- */
    for (int dev : devices) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaDeviceReset());
    }

    if (!cfg.json_output) printf("  测试完成!\n\n");
    return 0;
}
