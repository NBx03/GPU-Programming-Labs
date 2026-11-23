// ЛР4. Умножение матриц на CUDA: сравнение подходов из Глав 6 и 7 с библиотекой cuBLAS.
//
// Реализованы:
//   1. matrixMult_naive  -- базовая версия в глобальной памяти (Глава 6 методички, стр. 60-64).
//   2. matrixMult_shared -- оптимизированная версия с тайлингом в __shared__ памяти (Глава 7 методички, стр. 71-74).
//   3. cuBLAS gemm       -- промышленная реализация от NVIDIA (cublasSgemm / cublasDgemm).
//
// Вычисляются:
//   - Время выполнения ядра (мс) через cudaEvent
//   - Производительность в GFLOPS: 2 * M * N * K / (time * 1e9)
//   - Максимальная разница с эталоном (проверка корректности)
//   - Ускорение относительно наивной версии

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/cuda_check.h"

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status_ = (call);                                         \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",           \
                         __FILE__, __LINE__, (int)status_);                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define BLOCK_SIZE 16

// ==========================================================================================
// 1. НАИВНОЕ УМНОЖЕНИЕ МАТРИЦ (Глава 6 методички, стр. 60-64)
// ==========================================================================================
// Каждый поток вычисляет один элемент результирующей матрицы C.
// Строка матрицы A и столбец матрицы B каждый раз заново читаются из глобальной памяти.
// Число чтений из глобальной памяти на 1 элемент C = 2 * K.
template <typename T>
__global__ void matrixMult_naive(const T* __restrict__ A, const T* __restrict__ B,
                                 T* __restrict__ C, int Acols, int Bcols, int Arows) {
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

// ==========================================================================================
// 2. ТАЙЛОВОЕ УМНОЖЕНИЕ В РАЗДЕЛЯЕМОЙ ПАМЯТИ (Глава 7 методички, стр. 71-74)
// ==========================================================================================
// Матрицы разбиваются на подматрицы (тайлы) размером BLOCK_SIZE x BLOCK_SIZE (16x16).
// Блок потоков 16x16 дружно загружает тайл A и тайл B в память __shared__,
// синхронизируется через __syncthreads(), перемножает их в быстрой памяти на кристалле,
// снова синхронизируется и переходит к следующему шагу.
// Число чтений из медленной глобальной памяти сокращается в BLOCK_SIZE раз (в 16 раз)!
template <typename T>
__global__ void matrixMult_shared(const T* __restrict__ A, const T* __restrict__ B,
                                  T* __restrict__ C, int Acols, int Bcols, int Arows) {
    // Индекс начала первой подматрицы A, которую обрабатывает блок
    int aBegin = Acols * blockDim.y * blockIdx.y;
    int aEnd   = aBegin + Acols - 1;
    int aStep  = blockDim.x;

    // Индекс начала первой подматрицы B, которую обрабатывает блок
    int bBegin = blockDim.x * blockIdx.x;
    int bStep  = blockDim.y * Bcols;

    // Выделение разделяемой памяти для тайлов
    __shared__ T as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T bs[BLOCK_SIZE][BLOCK_SIZE];

    T sum = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        // Загрузка подматрицы A в разделяемую память (с проверкой границ)
        int a_curr_col = (ia - aBegin) + threadIdx.x;
        if (row < Arows && a_curr_col < Acols) {
            as[threadIdx.y][threadIdx.x] = A[ia + Acols * threadIdx.y + threadIdx.x];
        } else {
            as[threadIdx.y][threadIdx.x] = 0;
        }

        // Загрузка подматрицы B в разделяемую память (с проверкой границ)
        int b_curr_row = (ib - bBegin) / Bcols + threadIdx.y;
        if (b_curr_row < Acols && col < Bcols) {
            bs[threadIdx.y][threadIdx.x] = B[ib + Bcols * threadIdx.y + threadIdx.x];
        } else {
            bs[threadIdx.y][threadIdx.x] = 0;
        }

        // Синхронизация 1: ждем, пока все нити загрузят элементы в разделяемую память
        __syncthreads();

        // Перемножение двух подматриц в быстрой памяти на кристалле
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        }

        // Синхронизация 2: ждем, пока все нити закончат вычисления с текущим тайлом,
        // перед тем как затирать его следующими данными
        __syncthreads();
    }

    if (row < Arows && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

// ==========================================================================================
// 3. ЭТАЛОН НА CPU (для небольших матриц и проверки корректности)
// ==========================================================================================
template <typename T>
void cpu_matrix_mult(const T* A, const T* B, T* C, int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ==========================================================================================
// СТРУКТУРЫ РЕЗУЛЬТАТОВ И ХОСТ-ОБВЯЗКА
// ==========================================================================================
struct BenchResult {
    std::string name;
    double time_ms;
    double gflops;
    double speedup;
    double max_diff;
};

// Запуск cuBLAS gemm
static void run_cublas_gemm(cublasHandle_t handle, int M, int K, int N,
                            const float* d_A, const float* d_B, float* d_C) {
    // cuBLAS использует Column-Major (по столбцам), а C/C++ массив лежит Row-Major (по строкам).
    // Математическое свойство: C_row = A_row * B_row  <=>  C^T_col = B^T_col * A^T_col.
    // Передавая B и A с переставленными размерами, мы получаем правильный Row-Major результат в C без транспонирования!
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

int main(int argc, char** argv) {
    int N = 1024; // размер матрицы NxN по умолчанию
    int reps = 5; // число повторов для усреднения
    bool only_kernel = false; // для Nsight profiling (1 прогон выбранного ядра)
    std::string target_kernel = "all";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (arg == "--kernel" && i + 1 < argc) target_kernel = argv[++i];
        else if (arg == "--help") {
            std::printf("Использование: %s [--size N] [--reps R] [--kernel naive|shared|cublas|all]\n", argv[0]);
            return 0;
        }
    }

    int M = N, K = N; // квадратные матрицы
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    std::printf("======================================================================\n");
    std::printf("ЛР4. Умножение матриц на CUDA: %d x %d (float), повторов: %d\n", N, N, reps);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Устройство: %s (Архитектура sm_%d%d, Память: %.1f ГБ)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("======================================================================\n\n");

    // Инициализация матриц на хосте случайными числами в диапазоне [-1, 1]
    std::vector<float> h_A((size_t)M * K);
    std::vector<float> h_B((size_t)K * N);
    std::vector<float> h_C_ref((size_t)M * N);
    std::vector<float> h_C_out((size_t)M * N);

    std::srand(42);
    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;

    // Выделение памяти на GPU
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Вычисляем эталон через cuBLAS
    run_cublas_gemm(cublas_handle, M, K, N, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Параметры сетки
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Общее число операций умножения и сложения: 2 * M * N * K
    double total_flops = 2.0 * (double)M * (double)N * (double)K;

    EventTimer timer;
    std::vector<BenchResult> results;

    // Функция проверки погрешности
    auto calc_diff = [&](const float* test_ptr) -> double {
        double max_d = 0.0;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            double d = std::abs((double)test_ptr[i] - (double)h_C_ref[i]);
            if (d > max_d) max_d = d;
        }
        return max_d;
    };

    // 1. НАИВНАЯ ВЕРСИЯ (Глава 6)
    if (target_kernel == "all" || target_kernel == "naive") {
        // Прогрев
        matrixMult_naive<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            matrixMult_naive<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
            float ms = timer.stop();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double med_time = times[times.size() / 2];
        double gflops = (total_flops / (med_time * 1e-3)) / 1e9;

        CUDA_CHECK(cudaMemcpy(h_C_out.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
        double diff = calc_diff(h_C_out.data());
        results.push_back({"Гл.6: Naive (Global)", med_time, gflops, 1.0, diff});
    }

    // 2. ТАЙЛОВАЯ ВЕРСИЯ В SHARED ПАМЯТИ (Глава 7)
    if (target_kernel == "all" || target_kernel == "shared") {
        // Прогрев
        matrixMult_shared<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            matrixMult_shared<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
            float ms = timer.stop();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double med_time = times[times.size() / 2];
        double gflops = (total_flops / (med_time * 1e-3)) / 1e9;

        CUDA_CHECK(cudaMemcpy(h_C_out.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
        double diff = calc_diff(h_C_out.data());
        double speedup = results.empty() ? 1.0 : (results[0].time_ms / med_time);
        results.push_back({"Гл.7: Tiled (Shared)", med_time, gflops, speedup, diff});
    }

    // 3. cuBLAS gemm
    if (target_kernel == "all" || target_kernel == "cublas") {
        // Прогрев
        run_cublas_gemm(cublas_handle, M, K, N, d_A, d_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            run_cublas_gemm(cublas_handle, M, K, N, d_A, d_B, d_C);
            float ms = timer.stop();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double med_time = times[times.size() / 2];
        double gflops = (total_flops / (med_time * 1e-3)) / 1e9;

        double speedup = results.empty() ? 1.0 : (results[0].time_ms / med_time);
        results.push_back({"cuBLAS: cublasSgemm", med_time, gflops, speedup, 0.0});
    }

    // ВЫВОД ИТОГОВОЙ ТАБЛИЦЫ
    std::printf("%-24s | %10s | %10s | %10s | %12s\n",
                "Реализация", "Время (мс)", "GFLOPS", "Ускорение", "Макс. разница");
    std::printf("-------------------------+------------+------------+------------+--------------\n");
    for (const auto& res : results) {
        std::printf("%-24s | %10.3f | %10.1f | %9.2fx | %12.4e\n",
                    res.name.c_str(), res.time_ms, res.gflops, res.speedup, res.max_diff);
    }
    std::printf("-------------------------+------------+------------+------------+--------------\n\n");

    // Очистка памяти
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
