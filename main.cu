//
// Created by 16937 on 2026/8/20.
//
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "LRU_cache.h"
#include "leetcode_h.h"

#if 0
int main() {
    LRU_cache<int, std::string> cache(1000, 16);

    cache.Put(1, "hello");
    cache.Put(2, "world");

    auto value = cache.Get(1);
    if (value.has_value()) {
        std::cout << *value << '\n';
    }

    cache.Erase(2);

    std::cout << cache.Size() << '\n';
}
#endif

#include "cuda_wrapper.h"

// gpu_gemm_v2 的定义在 src/cuda_wrapper.cu,声明见 include/cuda_wrapper.h。
// 这里只保留 host 侧的测试代码。TILE 与 cuda_wrapper.cu 保持一致(block 必须 (TILE, TILE))。
#define TILE 32
// gpu_gemm_v3 的配置(block 输出 64x64、每线程 4x4),与 src/cuda_wrapper.cu 保持一致
#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4

// CPU 参考实现:三层循环,逐个输出元素累加
static void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU 与 CPU 结果对比。浮点累加顺序不同会有微小误差,用相对误差容限。
static bool gemm_check(const float* got, const float* want, int M, int N) {
    constexpr float tol = 1e-3f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float d = std::fabs(got[i * N + j] - want[i * N + j]);
            const float w = std::fabs(want[i * N + j]);
            if (d > tol * std::max(1.0f, w)) {
                std::cerr << "gemm mismatch at (" << i << ',' << j << "): "
                          << "gpu=" << got[i * N + j] << " cpu=" << want[i * N + j] << '\n';
                return false;
            }
        }
    }
    return true;
}

using GemmKernel = void (*)(const float*, const float*, float*, int, int, int);

// 启动一个内核、拷回结果并与 CPU 参考比较
static bool gemm_run_and_check(GemmKernel kernel, dim3 grid, dim3 block,
                               const float* dA, const float* dB, float* dC,
                               const float* C_ref, int M, int N, int K,
                               const char* name) {
    std::vector<float> C(M * N);
    kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << name << " launch failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    err = cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << name << " D2H copy failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    const bool ok = gemm_check(C.data(), C_ref, M, N);
    std::cout << name << ": " << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

// 用 CUDA event 测内核平均耗时并打印 TFLOPS
static void gemm_bench(GemmKernel kernel, dim3 grid, dim3 block,
                       const float* dA, const float* dB, float* dC,
                       int M, int N, int K, int iters, const char* name) {
    kernel<<<grid, block>>>(dA, dB, dC, M, N, K);   // 预热
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double tflops = 2.0 * M * N * K / (ms / iters * 1e-3) / 1e12;
    std::cout << name << ": " << (ms / iters) << " ms  (" << tflops << " TFLOPS)\n";
}

static void test_gemm() {
    // ---- 正确性:小尺寸,且故意不取 block 输出尺寸的倍数,逼出越界补 0 的分支 ----
    const int M = 100, N = 120, K = 100;
    std::vector<float> A(M * K), B(K * N), C_ref(M * N);

    // 确定性伪随机数,每次运行结果一致,方便复现
    unsigned seed = 12345u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xffff) / 256.0f - 0.5f;   // [-0.5, 0.5)
    };
    for (auto& x : A) x = next();
    for (auto& x : B) x = next();
    gemm_cpu(A.data(), B.data(), C_ref.data(), M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err = cudaMalloc(&dA, A.size() * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&dB, B.size() * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    if (err == cudaSuccess) err = cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "gemm alloc/copy failed: " << cudaGetErrorString(err) << '\n';
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return;
    }

    dim3 block2(TILE, TILE);
    dim3 grid2((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block3(BN / TN, BM / TM);               // 16x16
    dim3 grid3((N + BN - 1) / BN, (M + BM - 1) / BM);

    std::cout << "gemm correctness (M=" << M << " N=" << N << " K=" << K << "):\n";
    gemm_run_and_check(gpu_gemm_v2, grid2, block2, dA, dB, dC, C_ref.data(), M, N, K, "  v2 tiling   ");
    gemm_run_and_check(gpu_gemm_v3, grid3, block3, dA, dB, dC, C_ref.data(), M, N, K, "  v3 reg-tile ");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    // ---- 性能:1024³,比较 v2 / v3 ----
    const int MB = 1024, NB = 1024, KB = 1024;
    std::vector<float> Ab(MB * KB), Bb(KB * NB);
    for (auto& x : Ab) x = next();
    for (auto& x : Bb) x = next();

    float *dbA = nullptr, *dbB = nullptr, *dbC = nullptr;
    if (cudaMalloc(&dbA, Ab.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dbB, Bb.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dbC, (size_t)MB * NB * sizeof(float)) != cudaSuccess) {
        std::cerr << "bench alloc failed\n";
        return;
    }
    cudaMemcpy(dbA, Ab.data(), Ab.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dbB, Bb.data(), Bb.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block2b(TILE, TILE);
    dim3 grid2b((NB + TILE - 1) / TILE, (MB + TILE - 1) / TILE);
    dim3 block3b(BN / TN, BM / TM);
    dim3 grid3b((NB + BN - 1) / BN, (MB + BM - 1) / BM);

    std::cout << "gemm bench (M=N=K=" << MB << "):\n";
    gemm_bench(gpu_gemm_v2, grid2b, block2b, dbA, dbB, dbC, MB, NB, KB, 20, "  v2 tiling   ");
    gemm_bench(gpu_gemm_v3, grid3b, block3b, dbA, dbB, dbC, MB, NB, KB, 20, "  v3 reg-tile ");

    cudaFree(dbA); cudaFree(dbB); cudaFree(dbC);
}

// ================= SoftMax 测试 =================

// CPU 参考：朴素逐行 softmax
static void softmax_cpu(const float* X, float* Y, int M, int N) {
    for (int r = 0; r < M; ++r) {
        const float* x = X + (size_t)r * N;
        float* y = Y + (size_t)r * N;
        float mx = x[0];
        for (int i = 1; i < N; ++i) mx = std::fmax(mx, x[i]);
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) sum += std::exp(x[i] - mx);
        const float inv = 1.0f / sum;
        for (int i = 0; i < N; ++i) y[i] = std::exp(x[i] - mx) * inv;
    }
}

// softmax 输出都在 (0,1]：逐元素比绝对值，顺带校验每行和为 1
static bool softmax_check(const float* got, const float* want, int M, int N) {
    for (int r = 0; r < M; ++r) {
        const float* g = got  + (size_t)r * N;
        const float* w = want + (size_t)r * N;
        float row_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(g[i] - w[i]) > 1e-5f) {
                std::cerr << "softmax mismatch at (" << r << ',' << i << "): got "
                          << g[i] << " want " << w[i] << '\n';
                return false;
            }
            row_sum += g[i];
        }
        if (std::fabs(row_sum - 1.0f) > 1e-3f) {
            std::cerr << "softmax row " << r << " sum = " << row_sum << '\n';
            return false;
        }
    }
    return true;
}

using SoftmaxKernel = void (*)(float*, float*, int, int);

// 启动内核并用 CUDA event 测平均耗时。v0/v1 签名相同，只差 grid/block/动态共享内存。
static void softmax_bench(SoftmaxKernel kernel, dim3 grid, dim3 block, size_t shmem,
                          float* dX, float* dY, int M, int N, int iters, const char* name) {
    kernel<<<grid, block, shmem>>>(dX, dY, M, N);   // 预热
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        kernel<<<grid, block, shmem>>>(dX, dY, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << name << ": " << (ms / iters) << " ms/iter\n";
}

static void test_softmax() {
    // ---- 正确性：N 取 1000，既不是 256 的倍数、单行又够大，逼出分片与越界分支 ----
    const int Mc = 1000, Nc = 1000;
    std::vector<float> X(Mc * Nc), Y_cpu(Mc * Nc), Y0(Mc * Nc), Y1(Mc * Nc), Y2(Mc * Nc);

    unsigned seed = 999u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xffff) / 256.0f;   // [0, ~64)
    };
    for (auto& v : X) v = next();
    softmax_cpu(X.data(), Y_cpu.data(), Mc, Nc);

    float *dX = nullptr, *dY0 = nullptr, *dY1 = nullptr, *dY2 = nullptr;
    cudaError_t err = cudaMalloc(&dX, X.size() * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&dY0, Y_cpu.size() * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&dY1, Y_cpu.size() * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&dY2, Y_cpu.size() * sizeof(float));
    if (err == cudaSuccess) err = cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "softmax alloc/copy failed: " << cudaGetErrorString(err) << '\n';
        cudaFree(dX); cudaFree(dY0); cudaFree(dY1); cudaFree(dY2);
        return;
    }

    const int BLK = 256;
    std::cout << "softmax correctness (M=" << Mc << " N=" << Nc << "):\n";

    // v0：一个线程一行，grid 覆盖 M 行
    softmax_v0<<<(Mc + BLK - 1) / BLK, BLK>>>(dX, dY0, Mc, Nc);
    // v1：一个 block 一行，gridDim.x = M，第三参传动态共享内存 N*4 字节
    softmax_v1<<<Mc, BLK, (size_t)Nc * sizeof(float)>>>(dX, dY1, Mc, Nc);
    // v2：一个 block 一行、流式，只用静态共享内存（N 不必 < 12k）
    softmax_v2<<<Mc, BLK>>>(dX, dY2, Mc, Nc);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "softmax launch failed: " << cudaGetErrorString(err) << '\n';
        return;
    }
    cudaMemcpy(Y0.data(), dY0, Y0.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y1.data(), dY1, Y1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y2.data(), dY2, Y2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  v0 one-thread/row : " << (softmax_check(Y0.data(), Y_cpu.data(), Mc, Nc) ? "PASS" : "FAIL") << '\n';
    std::cout << "  v1 one-block/row  : " << (softmax_check(Y1.data(), Y_cpu.data(), Mc, Nc) ? "PASS" : "FAIL") << '\n';
    std::cout << "  v2 online/stream  : " << (softmax_check(Y2.data(), Y_cpu.data(), Mc, Nc) ? "PASS" : "FAIL") << '\n';
    cudaFree(dX); cudaFree(dY0); cudaFree(dY1); cudaFree(dY2);

    // ---- 性能：M 行 × N 列；v0 只有 M 个线程可用，v1 有 M×blockDim 个 ----
    const int Mb = 2048, Nb = 4096;
    std::vector<float> Xb(Mb * Nb);
    for (auto& v : Xb) v = next();

    float *dbX = nullptr, *dbY = nullptr;
    if (cudaMalloc(&dbX, Xb.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dbY, Xb.size() * sizeof(float)) != cudaSuccess) {
        std::cerr << "softmax bench alloc failed\n";
        return;
    }
    cudaMemcpy(dbX, Xb.data(), Xb.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "softmax bench (M=" << Mb << " N=" << Nb << "):\n";
    softmax_bench(softmax_v0, dim3((Mb + BLK - 1) / BLK), dim3(BLK), 0,
                  dbX, dbY, Mb, Nb, 10, "  v0 one-thread/row");
    softmax_bench(softmax_v1, dim3(Mb), dim3(BLK), (size_t)Nb * sizeof(float),
                  dbX, dbY, Mb, Nb, 10, "  v1 one-block/row ");
    softmax_bench(softmax_v2, dim3(Mb), dim3(BLK), 0,
                  dbX, dbY, Mb, Nb, 10, "  v2 online/stream");

    cudaFree(dbX); cudaFree(dbY);
}

// 超长行：N=65536 > 48KB 共享内存上限（N*4≈12288），v1 的动态共享内存版跑不了。
// 这里只有流式的 v2 能上场，验证它对"行长度无上限"这一承诺。
static void test_softmax_long() {
    const int Ml = 128, Nl = 65536;
    std::vector<float> Xl(Ml * Nl), Yl(Ml * Nl);

    unsigned seed = 777u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xffff) / 256.0f;
    };
    for (auto& v : Xl) v = next();

    float *dX = nullptr, *dY = nullptr;
    if (cudaMalloc(&dX, Xl.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dY, Xl.size() * sizeof(float)) != cudaSuccess) {
        std::cerr << "long-row alloc failed\n";
        return;
    }
    cudaMemcpy(dX, Xl.data(), Xl.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "softmax long-row (M=" << Ml << " N=" << Nl
              << ", N > 48KB 共享内存上限 → 只有 v2 能跑):\n";
    softmax_bench(softmax_v2, dim3(Ml), dim3(256), 0,
                  dX, dY, Ml, Nl, 20, "  v2 online/stream");

    // 验证长行下结果仍正确（行和 = 1）
    cudaMemcpy(Yl.data(), dY, Yl.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int r = 0; r < Ml; ++r) {
        const float* g = Yl.data() + (size_t)r * Nl;
        float s = 0.0f;
        for (int i = 0; i < Nl; ++i) s += g[i];
        if (std::fabs(s - 1.0f) > 1e-3f) { ok = false; std::cerr << "row " << r << " sum=" << s << '\n'; }
    }
    std::cout << "  long-row row-sum check: " << (ok ? "PASS" : "FAIL") << '\n';

    cudaFree(dX); cudaFree(dY);
}

int main() {
    test_warp_reduce<<<1, 32>>>();

    // 内核启动是异步的,必须等它执行完,否则 main 一返回进程就退出了,
    // device printf 的输出也会被丢掉(之前的 exit 0 但没有任何输出就是这个原因)。
    cudaDeviceSynchronize();
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
        sum += i;
    }
    std::cout << "sum =" << sum << '\n';

    test_gemm();

    test_softmax();
    test_softmax_long();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    return 0;
}


