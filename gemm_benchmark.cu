/**
 * gemm_benchmark.cu  —  NVIDIA A100 GEMM 算力基准测试 (v3 最终优化版)
 *
 * 核心策略:
 *   1. cuBLASLt + heuristic 算法选择 (确保最优 kernel)
 *   2. 单流 cuBLASLt (避免多流竞争 SM 资源)
 *   3. 功耗压力测试 (大矩阵 + 长时间运行拉满功耗)
 *   4. nvidia-smi 自动锁定最高性能模式 (400W TDP)
 *   5. 预分配显存 (消除分配开销)
 *
 * 精度: FP64 / FP32 / TF32 / FP16 / BF16 / INT8 / FP8(模拟)
 *
 * 用法: ./gemm_benchmark [矩阵大小] [迭代次数]
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
static constexpr size_t WORKSPACE_SIZE = 128ULL * 1024 * 1024;  // 128 MB
static constexpr int    TDP_WATTS      = 400;                    // A100-SXM4 TDP

/* ================================================================
 *  GPU 性能模式 (nvidia-smi)
 * ================================================================ */
static std::string get_pci_bus_id(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    char buf[32];
    snprintf(buf, sizeof(buf), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    return buf;
}

static int run_silent(const char *cmd) {
    return WEXITSTATUS(system(cmd));
}

static void setup_gpu_performance(const std::string &busId, int dev) {
    printf("  [GPU 调优] 设置最高性能模式...\n");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    char cmd[512];
    // 持久模式
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -pm 1 2>/dev/null", busId.c_str());
    run_silent(cmd);

    // 锁定最高 SM 时钟 (A100 boost = 1410 MHz)
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --lock-gpu-clocks=%d,%d 2>/dev/null",
             busId.c_str(), prop.clockRate / 1000, prop.clockRate / 1000);
    run_silent(cmd);

    // 设置最高应用时钟
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --applications-clocks=%d,%d 2>/dev/null",
             busId.c_str(), prop.memoryClockRate / 1000, prop.clockRate / 1000);
    run_silent(cmd);

    // 设置最高功率限制 (A100-SXM4 = 400W)
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --power-limit=%d 2>/dev/null",
             busId.c_str(), TDP_WATTS);
    run_silent(cmd);

    // 查询当前状态
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --query-gpu=clocks.sm,power.limit,pstate "
             "--format=csv,noheader 2>/dev/null", busId.c_str());
    printf("  [GPU 调优] 当前状态: ");
    fflush(stdout);
    run_silent(cmd);
    printf("  [GPU 调优] 完成 (Bus ID: %s)\n\n", busId.c_str());
}

static void restore_gpu(const std::string &busId) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -rgc 2>/dev/null", busId.c_str());
    run_silent(cmd);
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -rac 2>/dev/null", busId.c_str());
    run_silent(cmd);
}

/* ================================================================
 *  工具函数
 * ================================================================ */
static inline double gemm_flops(int M, int N, int K) {
    return 2.0 * M * N * K;
}

static void init_float(float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}
static void init_double(double *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

static void print_result(const char *label, int N, double gflops, double ms, int iters) {
    printf("  %-8s  %5d x %5d  |  %9.1f GFLOPS  (%7.2f TFLOPS)  |  %7.2f ms  (%d iters)\n",
           label, N, N, gflops, gflops / 1000.0, ms, iters);
}

/* ================================================================
 *  cuBLASLt GEMM — 单次执行 (heuristic 算法选择)
 * ================================================================ */
static cublasStatus_t run_cublaslt_gemm(
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace, size_t ws_size,
    int N,
    cudaDataType_t a_type, cudaDataType_t b_type,
    cudaDataType_t c_type, cublasComputeType_t compute_type,
    const void *alpha, const void *beta,
    const void *d_A, const void *d_B, void *d_C)
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t results[3];
    int count = 0;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    status = cublasLtMatmulDescCreate(&opDesc, compute_type, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto done;

    // FP64 需要 FP64 scale type
    if (compute_type == CUBLAS_COMPUTE_64F) {
        cudaDataType_t st = CUDA_R_64F;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                       &st, sizeof(st));
    }

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, a_type, N, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, b_type, N, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, c_type, N, N, N));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    // 搜索 top-3 算法，选最快的
    status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
        pref, 3, results, &count);

    if (status != CUBLAS_STATUS_SUCCESS || count == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        goto done;
    }

    // 使用最优算法
    status = cublasLtMatmul(ltHandle, opDesc,
                            alpha, d_A, Adesc, d_B, Bdesc,
                            beta, d_C, Cdesc, d_C, Cdesc,
                            &results[0].algo, workspace, ws_size, stream);

done:
    if (pref)    cublasLtMatmulPreferenceDestroy(pref);
    if (Adesc)   cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc)   cublasLtMatrixLayoutDestroy(Bdesc);
    if (Cdesc)   cublasLtMatrixLayoutDestroy(Cdesc);
    if (opDesc)  cublasLtMatmulDescDestroy(opDesc);
    return status;
}

/* ================================================================
 *  单流 GEMM 基准测试 (cuBLASLt)
 * ================================================================ */
static void bench_gemm(
    const char *label,
    int N, int warmup, int iters,
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace,
    cudaDataType_t a_type, cudaDataType_t b_type,
    cudaDataType_t c_type, cublasComputeType_t compute_type,
    const void *alpha, const void *beta,
    void *d_A, void *d_B, void *d_C)
{
    // 暖机
    for (int w = 0; w < warmup; w++) {
        cublasStatus_t st = run_cublaslt_gemm(
            ltHandle, stream, workspace, WORKSPACE_SIZE,
            N, a_type, b_type, c_type, compute_type,
            alpha, beta, d_A, d_B, d_C);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("  %-8s  %5d x %5d  |  *** cuBLASLt 不支持此配置 (status=%d) ***\n",
                   label, N, N, (int)st);
            return;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        run_cublaslt_gemm(
            ltHandle, stream, workspace, WORKSPACE_SIZE,
            N, a_type, b_type, c_type, compute_type,
            alpha, beta, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = total_ms / iters;
    double gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);

    print_result(label, N, gflops, avg_ms, iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/* ================================================================
 *  FP8 GEMM (自定义内核, A100 模拟)
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

static void bench_fp8(int N, int warmup, int iters, cudaStream_t stream) {
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

    // 初始化
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
    double gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);

    print_result("FP8*", N, gflops, avg_ms, iters);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dA16); cudaFree(dB16);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

/* ================================================================
 *  功耗压力测试 — 大矩阵长时间运行
 * ================================================================ */
static void power_stress_test(
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace,
    void *d_A, void *d_B, void *d_C,
    int N, double duration_sec)
{
    printf("  [功耗压力] 运行 FP16 %dx%d GEMM %.0f 秒...\n", N, N, duration_sec);

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
    double tflops = gemm_flops(N, N, N) * count / (elapsed * 1e12);

    printf("  [功耗压力] 完成: %d 次 GEMM, %.2f 秒, 平均 %.2f TFLOPS (%.1f GFLOPS)\n\n",
           count, elapsed, tflops, tflops * 1000.0);
}

/* ================================================================
 *  读取当前功耗
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

/* ================================================================
 *  主函数
 * ================================================================ */
int main(int argc, char *argv[]) {
    int N_custom = 0;
    int iters = 20;

    if (argc >= 2 && atoi(argv[1]) > 0) N_custom = atoi(argv[1]);
    if (argc >= 3 && atoi(argv[2]) > 0) iters = atoi(argv[2]);

    std::vector<int> sizes;
    if (N_custom > 0) {
        sizes = {N_custom};
    } else {
        sizes = {4096, 8192, 16384};
    }
    int warmup = 10;

    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::string busId = get_pci_bus_id(dev);

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  NVIDIA A100 GEMM 算力基准测试 v3 — cuBLASLt + 功耗优化                ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  GPU:              %s\n", prop.name);
    printf("  Compute Cap:      %d.%d\n", prop.major, prop.minor);
    printf("  SM Count:         %d\n", prop.multiProcessorCount);
    printf("  Clock (max):      %d MHz\n", prop.clockRate / 1000);
    printf("  Memory:           %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  CUDA Version:     %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("\n");

    setup_gpu_performance(busId, dev);

    printf("  基准测试参数:\n");
    printf("    矩阵尺寸:       ");
    for (int s : sizes) printf("%d ", s);
    printf("\n");
    printf("    迭代次数:       %d\n", iters);
    printf("    暖机次数:       %d\n", warmup);
    printf("    工作区:         %zu MB\n", WORKSPACE_SIZE / (1024 * 1024));
    printf("\n");

    /* ---------- 初始化 cuBLASLt ---------- */
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void *workspace;
    CUDA_CHECK(cudaMalloc(&workspace, WORKSPACE_SIZE));

    /* ---------- 预分配显存 ---------- */
    int N_max = *std::max_element(sizes.begin(), sizes.end());
    size_t max_elem = (size_t)N_max * N_max;

    printf("  [内存] 预分配 buffer (%.1f GB for FP64 %dx%d)...\n",
           max_elem * 8.0 / (1024.0*1024.0*1024.0), N_max, N_max);

    void *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, max_elem * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, max_elem * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, max_elem * sizeof(double)));

    // 初始化
    {
        double *hd = (double *)malloc(max_elem * sizeof(double));
        init_double(hd, max_elem);
        CUDA_CHECK(cudaMemcpy(d_A, hd, max_elem * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, hd, max_elem * sizeof(double), cudaMemcpyHostToDevice));
        free(hd);
    }
    printf("\n");

    /* ============================================================
     *  逐尺寸、逐精度基准测试
     * ============================================================ */
    for (int N : sizes) {
        printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
        printf("  │  矩阵: %5d x %5d    FLOPs/GEMM: %.3e                               │\n",
               N, N, gemm_flops(N, N, N));
        printf("  └─────────────────────────────────────────────────────────────────────────┘\n");
        printf("  %-8s  %-14s  |  %-38s  |  %-12s\n",
               "Type", "Size", "Performance", "Time");
        printf("  ────────────────────────────────────────────────────────────────────────────────────────\n");

        // === FP64 ===
        {
            double alpha = 1.0, beta = 0.0;
            bench_gemm("FP64", N, warmup, iters, ltHandle, stream, workspace,
                       CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F,
                       &alpha, &beta, d_A, d_B, d_C);
        }

        // === FP32 ===
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm("FP32", N, warmup, iters, ltHandle, stream, workspace,
                       CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                       &alpha, &beta, d_A, d_B, d_C);
        }

        // === TF32 ===
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm("TF32", N, warmup, iters, ltHandle, stream, workspace,
                       CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                       &alpha, &beta, d_A, d_B, d_C);
        }

        // === FP16 (FP32 累加) ===
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm("FP16", N, warmup, iters, ltHandle, stream, workspace,
                       CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F,
                       &alpha, &beta, d_A, d_B, d_C);
        }

        // === BF16 ===
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm("BF16", N, warmup, iters, ltHandle, stream, workspace,
                       CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
                       &alpha, &beta, d_A, d_B, d_C);
        }

        // === INT8 ===
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm("INT8", N, warmup, iters, ltHandle, stream, workspace,
                       CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I,
                       &alpha, &beta, d_A, d_B, d_C);
        }

        // === FP8 (自定义内核) ===
        bench_fp8(N, warmup, iters, stream);

        // 每组结束后读功耗
        read_power(busId);
        printf("\n");
    }

    /* ============================================================
     *  功耗压力测试 — 用最大矩阵跑 15 秒拉满功耗
     * ============================================================ */
    printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  功耗压力测试 (FP16 GEMM 持续满载 15 秒)                               │\n");
    printf("  └─────────────────────────────────────────────────────────────────────────┘\n");

    int N_stress = sizes.back();
    power_stress_test(ltHandle, stream, workspace, d_A, d_B, d_C,
                      N_stress, 15.0);

    // 读取功耗峰值
    printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  功耗读数                                                               │\n");
    printf("  └─────────────────────────────────────────────────────────────────────────┘\n");
    read_power(busId);
    printf("\n");

    /* ============================================================
     *  说明
     * ============================================================ */
    printf("  ╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                           说明 (Notes)                                 ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("  ║ 所有精度均使用 cuBLASLt + heuristic 算法选择 (top-3 中选最优)           ║\n");
    printf("  ║ FP64  : 双精度, Tensor Core (19.5 TFLOPS peak)                         ║\n");
    printf("  ║ FP32  : 单精度, CUDA Core (19.5 TFLOPS peak)                           ║\n");
    printf("  ║ TF32  : Tensor Float 32 (156 TFLOPS peak)                              ║\n");
    printf("  ║ FP16  : 半精度 Tensor Core, FP32 累加 (312 TFLOPS peak)               ║\n");
    printf("  ║ BF16  : Brain Float 16 Tensor Core (312 TFLOPS peak)                   ║\n");
    printf("  ║ INT8  : 8位整数 Tensor Core (624 TOPS peak)                            ║\n");
    printf("  ║ FP8*  : FP8 存储+FP16 计算 (模拟, A100 不原生支持)                     ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("  ║ A100-SXM4-40GB 官方 Dense 峰值:                                        ║\n");
    printf("  ║  FP64=9.7  FP32=19.5  TF32=156  FP16=312  BF16=312  INT8=624          ║\n");
    printf("  ║ (单位 TFLOPS/TOPS, 2:4 稀疏加速为上述值的 2 倍)                        ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════╝\n\n");

    /* ---------- 清理 ---------- */
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(workspace);
    cudaStreamDestroy(stream);
    cublasLtDestroy(ltHandle);

    restore_gpu(busId);
    CUDA_CHECK(cudaDeviceReset());

    printf("  测试完成!\n\n");
    return 0;
}
