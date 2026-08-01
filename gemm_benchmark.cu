/**
 * gemm_benchmark.cu  —  NVIDIA A100 GEMM 算力基准测试 (优化版)
 *
 * 核心优化:
 *   1. cuBLASLt + heuristic 算法选择 (确保最优 kernel)
 *   2. 多 CUDA Stream 并发执行 (饱和所有 SM，拉满功耗)
 *   3. nvidia-smi 自动锁定最高性能模式
 *   4. 预分配显存 (消除分配开销)
 *   5. 更大矩阵 + 更多迭代 (持续满载)
 *
 * 精度: FP64 / FP32 / TF32 / FP16 / BF16 / INT8 / FP8(模拟)
 *
 * 编译: make
 * 运行: ./gemm_benchmark [矩阵大小] [迭代次数] [流数量]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>

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
        cublasStatus_t _cublas_st = (call);                                    \
        if (_cublas_st != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLAS Error [%s:%d] status=%d\n", __FILE__,      \
                    __LINE__, (int)_cublas_st);                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ================================================================
 *  全局配置
 * ================================================================ */
static constexpr size_t WORKSPACE_SIZE = 128ULL * 1024 * 1024; // 128 MB / stream

/* ================================================================
 *  GPU 性能模式 (nvidia-smi)
 * ================================================================ */
static void setup_gpu_performance(int dev) {
    printf("  [GPU 调优] 设置最高性能模式...\n");

    // 获取 PCI Bus ID
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    char pciBusId[32];
    snprintf(pciBusId, sizeof(pciBusId), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    auto run_cmd = [&](const char *cmd) {
        int ret = system(cmd);
        if (ret != 0) {
            printf("  [GPU 调优] 警告: 命令失败 (可能需要 root): %s\n", cmd);
        }
    };

    char cmd[512];

    // 1. 持久模式 (保持驱动加载，减少首次调用延迟)
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -pm 1 2>/dev/null", pciBusId);
    run_cmd(cmd);

    // 2. 锁定最高 SM 时钟
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --lock-gpu-clocks=%d,%d 2>/dev/null",
             pciBusId, prop.clockRate / 1000, prop.clockRate / 1000);
    run_cmd(cmd);

    // 3. 设置最高应用时钟
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --applications-clocks=%d,%d 2>/dev/null",
             pciBusId,
             prop.memoryClockRate / 1000,
             prop.clockRate / 1000);
    run_cmd(cmd);

    // 4. 设置最高功率限制
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi -i %s --power-limit=%d 2>/dev/null",
             pciBusId, (int)(prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) > 80 ? 400 : 300);
    run_cmd(cmd);

    printf("  [GPU 调优] 完成 (Bus ID: %s)\n\n", pciBusId);
}

static void restore_gpu(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    char pciBusId[32];
    snprintf(pciBusId, sizeof(pciBusId), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -rgc 2>/dev/null", pciBusId);
    (void)system(cmd);
    snprintf(cmd, sizeof(cmd), "nvidia-smi -i %s -rac 2>/dev/null", pciBusId);
    (void)system(cmd);
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
static void init_int8(int8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (int8_t)(rand() % 200 - 100);
}
static void init_half(__half *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
}
static void init_bf16(__nv_bfloat16 *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = __float2bfloat16((float)rand() / RAND_MAX * 2.0f - 1.0f);
}

static void print_result(const char *label, int N, double gflops, double ms, int iters) {
    printf("  %-8s  %5d x %5d  |  %9.1f GFLOPS  (%7.2f TFLOPS)  |  %7.2f ms  (%d iters x %d streams)\n",
           label, N, N, gflops, gflops / 1000.0, ms, iters, 0);
}

static void print_result_s(const char *label, int N, double gflops, double ms,
                           int iters, int nstreams) {
    printf("  %-8s  %5d x %5d  |  %9.1f GFLOPS  (%7.2f TFLOPS)  |  %7.2f ms  (%d iters x %d streams)\n",
           label, N, N, gflops, gflops / 1000.0, ms, iters, nstreams);
}

/* ================================================================
 *  cuBLASLt GEMM — 单 stream 单次执行
 *  支持: FP64, FP32, TF32, FP16, BF16, INT8
 * ================================================================ */
static cublasStatus_t run_cublaslt_gemm(
    cublasLtHandle_t ltHandle, cudaStream_t stream, void *workspace,
    size_t ws_size,
    int N,
    cudaDataType_t a_type, cudaDataType_t b_type,
    cudaDataType_t c_type, cublasComputeType_t compute_type,
    const void *alpha, const void *beta,
    const void *d_A, const void *d_B, void *d_C)
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t result;
    int returnedAlgoCount = 0;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    // 创建 matmul 描述
    status = cublasLtMatmulDescCreate(&opDesc, compute_type, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    // 对于 FP64, scale type 应为 CUDA_R_64F
    if (compute_type == CUBLAS_COMPUTE_64F) {
        cudaDataType_t scale_type = CUDA_R_64F;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                       &scale_type, sizeof(scale_type));
    }

    // 创建矩阵布局 (列主序)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, a_type, N, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, b_type, N, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, c_type, N, N, N));

    // 算法偏好设置
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    // 获取最优算法
    status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
        pref, 1, &result, &returnedAlgoCount);

    if (status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        goto cleanup;
    }

    // 执行 GEMM
    status = cublasLtMatmul(ltHandle, opDesc,
                            alpha, d_A, Adesc, d_B, Bdesc,
                            beta, d_C, Cdesc, d_C, Cdesc,
                            &result.algo, workspace, ws_size, stream);

cleanup:
    if (pref)    cublasLtMatmulPreferenceDestroy(pref);
    if (Adesc)   cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc)   cublasLtMatrixLayoutDestroy(Bdesc);
    if (Cdesc)   cublasLtMatrixLayoutDestroy(Cdesc);
    if (opDesc)  cublasLtMatmulDescDestroy(opDesc);
    return status;
}

/* ================================================================
 *  多流并发 GEMM 基准测试
 * ================================================================ */
static void bench_gemm_multistream(
    const char *label,
    int N, int warmup, int iters, int nstreams,
    cublasLtHandle_t *ltHandles, cudaStream_t *streams, void **workspaces,
    cudaDataType_t a_type, cudaDataType_t b_type,
    cudaDataType_t c_type, cublasComputeType_t compute_type,
    const void *alpha, const void *beta,
    void **d_A, void **d_B, void **d_C)
{
    size_t ws = WORKSPACE_SIZE;

    // 暖机 (所有流)
    for (int w = 0; w < warmup; w++) {
        for (int s = 0; s < nstreams; s++) {
            cublasStatus_t st = run_cublaslt_gemm(
                ltHandles[s], streams[s], workspaces[s], ws,
                N, a_type, b_type, c_type, compute_type,
                alpha, beta, d_A[s], d_B[s], d_C[s]);
            if (st != CUBLAS_STATUS_SUCCESS) {
                printf("  %-8s  %5d x %5d  |  *** 不支持此配置 (status=%d) ***\n",
                       label, N, N, (int)st);
                return;
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时: 所有流并发执行 iters 次
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        for (int s = 0; s < nstreams; s++) {
            run_cublaslt_gemm(
                ltHandles[s], streams[s], workspaces[s], ws,
                N, a_type, b_type, c_type, compute_type,
                alpha, beta, d_A[s], d_B[s], d_C[s]);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = total_ms / iters;

    // 总 FLOPs = 单 GEMM FLOPs (多流并发不增加 FLOPs，只增加吞吐)
    double gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);

    print_result_s(label, N, gflops, avg_ms, iters, nstreams);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/* ================================================================
 *  FP8 GEMM (自定义内核, A100 模拟) — 多流版
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

static void bench_fp8_multistream(int N, int warmup, int iters, int nstreams,
                                   cudaStream_t *streams) {
    constexpr int TILE = 32;
    size_t n_elem = (size_t)N * N;
    size_t fp8_sz = n_elem * sizeof(__nv_fp8_e4m3);
    size_t f16_sz = n_elem * sizeof(__half);

    // 每流分配独立 buffer
    std::vector<__nv_fp8_e4m3 *> dA(nstreams), dB(nstreams);
    std::vector<__half *> dC(nstreams);

    for (int s = 0; s < nstreams; s++) {
        CUDA_CHECK(cudaMalloc(&dA[s], fp8_sz));
        CUDA_CHECK(cudaMalloc(&dB[s], fp8_sz));
        CUDA_CHECK(cudaMalloc(&dC[s], f16_sz));
    }

    // 初始化数据
    __half *h = (__half *)malloc(f16_sz);
    init_half(h, n_elem);
    for (int s = 0; s < nstreams; s++) {
        __half *dA16, *dB16;
        CUDA_CHECK(cudaMalloc(&dA16, f16_sz));
        CUDA_CHECK(cudaMalloc(&dB16, f16_sz));
        CUDA_CHECK(cudaMemcpy(dA16, h, f16_sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB16, h, f16_sz, cudaMemcpyHostToDevice));
        int t = 256, b = (n_elem + t - 1) / t;
        f16_to_fp8_kernel<<<b, t, 0, streams[s]>>>(dA16, dA[s], n_elem);
        f16_to_fp8_kernel<<<b, t, 0, streams[s]>>>(dB16, dB[s], n_elem);
        cudaFree(dA16); cudaFree(dB16);
    }
    free(h);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 block(TILE, TILE);

    // 暖机
    for (int w = 0; w < warmup; w++) {
        for (int s = 0; s < nstreams; s++) {
            fp8_gemm_kernel<TILE><<<grid, block, 0, streams[s]>>>(
                dA[s], dB[s], dC[s], N, N, N);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        for (int s = 0; s < nstreams; s++) {
            fp8_gemm_kernel<TILE><<<grid, block, 0, streams[s]>>>(
                dA[s], dB[s], dC[s], N, N, N);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = total_ms / iters;
    double gflops = gemm_flops(N, N, N) / (avg_ms * 1e6);

    print_result_s("FP8*", N, gflops, avg_ms, iters, nstreams);

    for (int s = 0; s < nstreams; s++) {
        cudaFree(dA[s]); cudaFree(dB[s]); cudaFree(dC[s]);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

/* ================================================================
 *  打印分隔线
 * ================================================================ */
static void print_sep() {
    printf("  ─────────────────────────────────────────────────────────────────────────────────────────────\n");
}

/* ================================================================
 *  主函数
 * ================================================================ */
int main(int argc, char *argv[]) {
    /* ---------- 参数解析 ---------- */
    int N_default = 0;    // 0 = 使用默认列表
    int iters = 20;
    int nstreams = 4;

    if (argc >= 2 && atoi(argv[1]) > 0) N_default = atoi(argv[1]);
    if (argc >= 3 && atoi(argv[2]) > 0) iters = atoi(argv[2]);
    if (argc >= 4 && atoi(argv[3]) > 0) nstreams = atoi(argv[3]);

    std::vector<int> sizes;
    if (N_default > 0) {
        sizes = {N_default};
    } else {
        sizes = {4096, 8192, 16384};
    }

    int warmup = 5;

    /* ---------- GPU 信息 & 性能模式 ---------- */
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║      NVIDIA A100 GEMM 算力基准测试 — 优化版 (Multi-Stream + cuBLASLt)   ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  GPU:              %s\n", prop.name);
    printf("  Compute Cap:      %d.%d\n", prop.major, prop.minor);
    printf("  SM Count:         %d\n", prop.multiProcessorCount);
    printf("  Clock (max):      %d MHz\n", prop.clockRate / 1000);
    printf("  Memory:           %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  CUDA Version:     %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("\n");

    setup_gpu_performance(dev);

    printf("  基准测试参数:\n");
    printf("    矩阵尺寸:       ");
    for (int s : sizes) printf("%d ", s);
    printf("\n");
    printf("    迭代次数:       %d\n", iters);
    printf("    并发流数:       %d\n", nstreams);
    printf("    暖机次数:       %d\n", warmup);
    printf("    工作区大小:     %zu MB / stream\n", WORKSPACE_SIZE / (1024 * 1024));
    printf("\n");

    /* ---------- 创建 cuBLASLt handles & streams ---------- */
    std::vector<cublasLtHandle_t> ltHandles(nstreams);
    std::vector<cudaStream_t> streams(nstreams);
    std::vector<void *> workspaces(nstreams);

    for (int s = 0; s < nstreams; s++) {
        CUBLAS_CHECK(cublasLtCreate(&ltHandles[s]));
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&workspaces[s], WORKSPACE_SIZE));
    }

    /* ---------- 预分配显存 (每流独立 buffer) ---------- */
    // 最大矩阵尺寸用于预分配
    int N_max = *std::max_element(sizes.begin(), sizes.end());
    size_t max_elem = (size_t)N_max * N_max;

    printf("  [内存] 预分配 %d 组 buffer (%.1f GB/组 for FP64)...\n\n",
           nstreams, max_elem * 8.0 / (1024.0*1024.0*1024.0));

    // 为每个流预分配最大 buffer (按最大类型 FP64 分配)
    std::vector<void *> d_A(nstreams), d_B(nstreams), d_C(nstreams);
    for (int s = 0; s < nstreams; s++) {
        CUDA_CHECK(cudaMalloc(&d_A[s], max_elem * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B[s], max_elem * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C[s], max_elem * sizeof(double)));
    }

    // 初始化数据 (用最大尺寸)
    {
        size_t f64_sz = max_elem * sizeof(double);
        size_t f32_sz = max_elem * sizeof(float);
        double *hd = (double *)malloc(f64_sz);
        float  *hf = (float  *)malloc(f32_sz);
        init_double(hd, max_elem);
        init_float(hf, max_elem);
        for (int s = 0; s < nstreams; s++) {
            CUDA_CHECK(cudaMemcpy(d_A[s], hd, f64_sz, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B[s], hf, f32_sz, cudaMemcpyHostToDevice));
        }
        free(hd); free(hf);
    }

    /* ============================================================
     *  对每个矩阵尺寸执行全精度基准测试
     * ============================================================ */
    for (int N : sizes) {
        size_t n_elem = (size_t)N * N;

        printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
        printf("  │  矩阵尺寸: %5d x %5d    FLOPs/GEMM: %.3e                          │\n",
               N, N, gemm_flops(N, N, N));
        printf("  └─────────────────────────────────────────────────────────────────────────┘\n");
        printf("  %-8s  %-14s  |  %-38s  |  %-20s\n",
               "Type", "Size", "Performance", "Timing");
        print_sep();

        /* === FP64 === */
        {
            double alpha = 1.0, beta = 0.0;
            bench_gemm_multistream("FP64", N, warmup, iters, nstreams,
                ltHandles.data(), streams.data(), workspaces.data(),
                CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F,
                &alpha, &beta, d_A.data(), d_B.data(), d_C.data());
        }

        /* === FP32 === */
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm_multistream("FP32", N, warmup, iters, nstreams,
                ltHandles.data(), streams.data(), workspaces.data(),
                CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                &alpha, &beta, d_A.data(), d_B.data(), d_C.data());
        }

        /* === TF32 === */
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm_multistream("TF32", N, warmup, iters, nstreams,
                ltHandles.data(), streams.data(), workspaces.data(),
                CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                &alpha, &beta, d_A.data(), d_B.data(), d_C.data());
        }

        /* === FP16 === */
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm_multistream("FP16", N, warmup, iters, nstreams,
                ltHandles.data(), streams.data(), workspaces.data(),
                CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F,
                &alpha, &beta, d_A.data(), d_B.data(), d_C.data());
        }

        /* === BF16 === */
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm_multistream("BF16", N, warmup, iters, nstreams,
                ltHandles.data(), streams.data(), workspaces.data(),
                CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
                &alpha, &beta, d_A.data(), d_B.data(), d_C.data());
        }

        /* === INT8 (cuBLASLt heuristic) === */
        {
            float alpha = 1.0f, beta = 0.0f;
            bench_gemm_multistream("INT8", N, warmup, iters, nstreams,
                ltHandles.data(), streams.data(), workspaces.data(),
                CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I,
                &alpha, &beta, d_A.data(), d_B.data(), d_C.data());
        }

        /* === FP8 (自定义内核) === */
        bench_fp8_multistream(N, warmup, iters, nstreams, streams.data());

        printf("\n");
    }

    /* ============================================================
     *  功耗峰值采样
     * ============================================================ */
    printf("  ┌─────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  功耗采样 (运行 10 秒 FP16 GEMM 并监控功耗)                            │\n");
    printf("  └─────────────────────────────────────────────────────────────────────────┘\n");

    // 后台启动 nvidia-smi 功耗监控
    {
        int N_pwr = sizes.back(); // 用最大矩阵
        float alpha = 1.0f, beta = 0.0f;

        // 持续跑 GEMM 10 秒
        auto t0 = std::chrono::steady_clock::now();
        int count = 0;
        while (true) {
            for (int s = 0; s < nstreams; s++) {
                run_cublaslt_gemm(
                    ltHandles[s], streams[s], workspaces[s], WORKSPACE_SIZE,
                    N_pwr,
                    CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F,
                    &alpha, &beta, d_A[s], d_B[s], d_C[s]);
            }
            count++;
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            if (elapsed >= 10.0) break;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 读取功耗
        char cmd[256];
        cudaDeviceProp p2;
        CUDA_CHECK(cudaGetDeviceProperties(&p2, dev));
        char bus[32];
        snprintf(bus, sizeof(bus), "%04x:%02x:%02x.0",
                 p2.pciDomainID, p2.pciBusID, p2.pciDeviceID);
        snprintf(cmd, sizeof(cmd),
                 "nvidia-smi -i %s --query-gpu=power.draw,power.limit,clocks.sm "
                 "--format=csv,noheader 2>/dev/null", bus);
        printf("  ");
        fflush(stdout);
        (void)system(cmd);
        printf("\n");
    }

    /* ============================================================
     *  说明 & A100 官方峰值
     * ============================================================ */
    printf("  ╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                           说明 (Notes)                                 ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("  ║ FP64  : 双精度, Tensor Core (cuBLASLt heuristic 算法)                  ║\n");
    printf("  ║ FP32  : 单精度, CUDA Core                                              ║\n");
    printf("  ║ TF32  : Tensor Float 32, A100 Tensor Core 加速                        ║\n");
    printf("  ║ FP16  : 半精度, Tensor Core (FP32 累加)                                ║\n");
    printf("  ║ BF16  : Brain Float 16, Tensor Core (FP32 累加)                       ║\n");
    printf("  ║ INT8  : 8位整数, Tensor Core (INT32 累加, cuBLASLt)                   ║\n");
    printf("  ║ FP8*  : FP8 存储+FP16计算 (模拟, A100 不原生支持)                      ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("  ║ A100-SXM4-40GB 官方峰值 (Dense / 2:4 Sparse):                          ║\n");
    printf("  ║   FP64:  9.7  / 19.5 TFLOPS       FP32: 19.5 TFLOPS                  ║\n");
    printf("  ║   TF32:  156  / 312  TFLOPS       FP16: 312 / 624 TFLOPS             ║\n");
    printf("  ║   BF16:  312  / 624  TFLOPS       INT8: 624 / 1248 TOPS              ║\n");
    printf("  ║                                                                        ║\n");
    printf("  ║ * Dense 峰值约为 Sparse 标称值的 50%%                                   ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /* ---------- 清理 ---------- */
    for (int s = 0; s < nstreams; s++) {
        cudaFree(d_A[s]); cudaFree(d_B[s]); cudaFree(d_C[s]);
        cudaFree(workspaces[s]);
        cudaStreamDestroy(streams[s]);
        cublasLtDestroy(ltHandles[s]);
    }

    restore_gpu(dev);
    CUDA_CHECK(cudaDeviceReset());

    printf("  测试完成!\n\n");
    return 0;
}
