#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/cuda_check.h"

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t st = (call);                                              \
        if (st != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define BLOCK_SIZE 16

// Ядро из главы 6 (наивное через глобальную память)
template <typename T>
__global__ void matrixMult_naive(const T* A, const T* B, T* C, int Acols, int Bcols, int Arows) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= Arows || col >= Bcols) return;

    int i0 = Acols * row;
    int j0 = col;

    T sum = 0;
    for (int k = 0; k < Acols; ++k) {
        sum += A[i0 + k] * B[k * Bcols + j0];
    }
    C[row * Bcols + col] = sum;
}

// Ядро из главы 7 (тайлинг в shared памяти 16x16)
template <typename T>
__global__ void matrixMult_shared(const T* A, const T* B, T* C, int Acols, int Bcols, int Arows) {
    int aBegin = Acols * blockDim.y * blockIdx.y;
    int aEnd   = aBegin + Acols - 1;
    int aStep  = blockDim.x;

    int bBegin = blockDim.x * blockIdx.x;
    int bStep  = blockDim.y * Bcols;

    __shared__ T as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T bs[BLOCK_SIZE][BLOCK_SIZE];

    T sum = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        int a_col = (ia - aBegin) + threadIdx.x;
        int b_row = (ib - bBegin) / Bcols + threadIdx.y;

        as[threadIdx.y][threadIdx.x] = (row < Arows && a_col < Acols) ? A[ia + Acols * threadIdx.y + threadIdx.x] : 0;
        bs[threadIdx.y][threadIdx.x] = (b_row < Acols && col < Bcols) ? B[ib + Bcols * threadIdx.y + threadIdx.x] : 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < Arows && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

// cuBLAS sgemm (учитываем row-major через B^T * A^T = C^T)
static void run_cublas(cublasHandle_t handle, int M, int K, int N,
                       const float* d_A, const float* d_B, float* d_C) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             d_B, N, d_A, K,
                             &beta, d_C, N));
}

struct Result {
    std::string name;
    double time_ms;
    double gflops;
    double speedup;
    double diff;
};

int main(int argc, char** argv) {
    int N = 1024;
    int reps = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    }

    int M = N, K = N;
    size_t bytes = (size_t)N * N * sizeof(float);
    double flops = 2.0 * (double)N * (double)N * (double)N;

    std::printf("Matrix: %dx%d (float), reps: %d\n", N, N, reps);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    std::vector<float> h_A((size_t)N * N);
    std::vector<float> h_B((size_t)N * N);
    std::vector<float> h_ref((size_t)N * N);
    std::vector<float> h_out((size_t)N * N);

    std::srand(12345);
    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = (float)std::rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = (float)std::rand() / RAND_MAX - 0.5f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Эталон через cuBLAS
    run_cublas(handle, M, K, N, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    EventTimer timer;
    std::vector<Result> results;

    auto get_diff = [&](const float* ptr) {
        double max_d = 0.0;
        for (size_t i = 0; i < (size_t)N * N; ++i) {
            double d = std::abs((double)ptr[i] - (double)h_ref[i]);
            if (d > max_d) max_d = d;
        }
        return max_d;
    };

    // 1. Naive (Гл. 6)
    {
        matrixMult_naive<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            matrixMult_naive<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
            times.push_back(timer.stop());
        }
        std::sort(times.begin(), times.end());
        double t = times[times.size() / 2];
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"Naive (Ch.6)", t, (flops / (t * 1e-3)) / 1e9, 1.0, get_diff(h_out.data())});
    }

    // 2. Tiled shared (Гл. 7)
    {
        matrixMult_shared<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            matrixMult_shared<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
            times.push_back(timer.stop());
        }
        std::sort(times.begin(), times.end());
        double t = times[times.size() / 2];
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        double sp = results[0].time_ms / t;
        results.push_back({"Shared (Ch.7)", t, (flops / (t * 1e-3)) / 1e9, sp, get_diff(h_out.data())});
    }

    // 3. cuBLAS
    {
        run_cublas(handle, M, K, N, d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            run_cublas(handle, M, K, N, d_A, d_B, d_C);
            times.push_back(timer.stop());
        }
        std::sort(times.begin(), times.end());
        double t = times[times.size() / 2];
        double sp = results[0].time_ms / t;
        results.push_back({"cuBLAS", t, (flops / (t * 1e-3)) / 1e9, sp, 0.0});
    }

    std::printf("%-15s | %10s | %10s | %9s | %12s\n", "Version", "Time (ms)", "GFLOPS", "Speedup", "Max diff");
    std::printf("----------------+------------+------------+-----------+-------------\n");
    for (const auto& r : results) {
        std::printf("%-15s | %10.3f | %10.1f | %8.2fx | %12.4e\n",
                    r.name.c_str(), r.time_ms, r.gflops, r.speedup, r.diff);
    }
    std::printf("----------------+------------+------------+-----------+-------------\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
