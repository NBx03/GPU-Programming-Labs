// ЛР1. Точность и скорость sin / sinf / __sinf на GPU.
//
// arr[i] = sin((i % 360) * Pi / 180) для N = 1e9 элементов (float и double).
// Массив копируется на CPU, где считается ошибка
//     err = sum_i |sin((i % 360) * Pi / 180) - arr[i]| / N.
//
// Варианты (float):
//   sin(double)->float : (float)sin((double)x)   -- в CUDA перегрузка sin(float) == sinf,
//                        поэтому "честный" sin считается через double
//   sinf               : sinf(x)                 -- точная float-функция
//   __sinf             : __sinf(x)               -- быстрая аппаратная интринсика (SFU)
// Вариант double: sin(double).
//
// Считаются ДВЕ ошибки:
//   err_true : относительно sin(i%360 * pi/180), посчитанного в double (то, что просит задание;
//              включает округление аргумента до float);
//   err_fn   : относительно sin((double)x_float), где x_float -- именно тот float-аргумент,
//              который получила функция (чистая ошибка самой функции).
//
// Массив может обрабатываться чанками (по умолчанию 2^28 элементов), результат от этого не
// зависит, а память GPU/CPU не переполняется. --full пытается взять весь массив целиком.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../common/cuda_check.h"

// pi / 180 как double-литерал: одно и то же значение на host и device.
#define DEG2RAD 0.017453292519943295

enum class Fn { SinViaDouble, Sinf, FastSinf };

template <Fn F>
__device__ __forceinline__ float eval_f(float x) {
    if (F == Fn::SinViaDouble) return (float)sin((double)x);
    if (F == Fn::Sinf) return sinf(x);
    return __sinf(x);
}

// base = offset % 360 (смещение чанка), i -- индекс внутри чанка. n < 2^32.
template <Fn F>
__global__ void fill_float(float* out, unsigned n, unsigned base) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = (double)((base + i) % 360u) * DEG2RAD;
    out[i] = eval_f<F>((float)a);
}

__global__ void fill_double(double* out, unsigned n, unsigned base) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sin((double)((base + i) % 360u) * DEG2RAD);
}

// Микро-бенчмарк "вычисления >> память": iters зависимых вызовов функции на элемент.
template <Fn F>
__global__ void bench_float(float* out, unsigned n, int iters) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (float)((double)(i % 360u) * DEG2RAD);
    float acc = x;
    for (int k = 0; k < iters; ++k) acc = eval_f<F>(acc) + x;
    out[i] = acc;
}

__global__ void bench_double(double* out, unsigned n, int iters) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = (double)(i % 360u) * DEG2RAD;
    double acc = x;
    for (int k = 0; k < iters; ++k) acc = sin(acc) + x;
    out[i] = acc;
}

// ------------------------------------------------------------------------------------------
// Ошибки на CPU
// ------------------------------------------------------------------------------------------
struct ErrAcc {
    double sum_true = 0, sum_fn = 0, max_true = 0, max_fn = 0;
};

template <class T>
static void accumulate_error(const T* buf, long long n, unsigned base, const double* ref_true,
                             const double* ref_fn, ErrAcc& acc) {
    double st = 0, sf = 0, mt = 0, mf = 0;
#pragma omp parallel for reduction(+ : st, sf) reduction(max : mt, mf) schedule(static)
    for (long long j = 0; j < n; ++j) {
        unsigned r = (unsigned)((base + (unsigned long long)j) % 360ull);
        double v = (double)buf[j];
        double et = std::fabs(ref_true[r] - v);
        double ef = std::fabs(ref_fn[r] - v);
        st += et;
        sf += ef;
        if (et > mt) mt = et;
        if (ef > mf) mf = ef;
    }
    acc.sum_true += st;
    acc.sum_fn += sf;
    acc.max_true = std::max(acc.max_true, mt);
    acc.max_fn = std::max(acc.max_fn, mf);
}

// ------------------------------------------------------------------------------------------
struct Options {
    long long n = 1000000000LL;
    long long chunk = 1LL << 28;
    bool full = false;
    int block = 256;
    int reps = 5;
    bool micro = true;
    long long micro_n = 1LL << 24;
    int micro_iters = 256;
    std::string csv;
};

struct Row {
    std::string type, fn;
    double err_true = 0, err_fn = 0, max_true = 0, max_fn = 0;
    double kernel_mean = 0, kernel_min = 0, d2h_mean = 0, total_mean = 0, gbps = 0;
};

static double mean_of(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
}
static double min_of(const std::vector<double>& v) {
    return v.empty() ? 0 : *std::min_element(v.begin(), v.end());
}

using Launch = std::function<void(void* d, unsigned n, unsigned base, int block)>;

static void launch_check() {
    CUDA_CHECK(cudaGetLastError());
}

template <class T>
static Row run_variant(const char* type, const char* fn, const Launch& launch, const Options& o,
                       long long chunk, void* d_buf, T* h_buf, const double* ref_true,
                       const double* ref_fn) {
    // прогрев (JIT, кэши, частоты)
    launch(d_buf, (unsigned)std::min<long long>(chunk, 1 << 20), 0, o.block);
    launch_check();
    CUDA_CHECK(cudaDeviceSynchronize());

    Row row;
    row.type = type;
    row.fn = fn;
    std::vector<double> kern, d2h;
    ErrAcc acc;
    EventTimer tk, tc;
    for (int rep = 0; rep < o.reps; ++rep) {
        double kms = 0, cms = 0;
        for (long long off = 0; off < o.n; off += chunk) {
            unsigned n = (unsigned)std::min<long long>(chunk, o.n - off);
            unsigned base = (unsigned)(off % 360);
            tk.start();
            launch(d_buf, n, base, o.block);
            kms += tk.stop();
            launch_check();
            tc.start();
            CUDA_CHECK(cudaMemcpy(h_buf, d_buf, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost));
            cms += tc.stop();
            if (rep == 0) accumulate_error(h_buf, n, base, ref_true, ref_fn, acc);
        }
        kern.push_back(kms);
        d2h.push_back(cms);
    }
    row.err_true = acc.sum_true / (double)o.n;
    row.err_fn = acc.sum_fn / (double)o.n;
    row.max_true = acc.max_true;
    row.max_fn = acc.max_fn;
    row.kernel_mean = mean_of(kern);
    row.kernel_min = min_of(kern);
    row.d2h_mean = mean_of(d2h);
    row.total_mean = row.kernel_mean + row.d2h_mean;
    row.gbps = (double)o.n * sizeof(T) / (row.kernel_min * 1e-3) / 1e9;
    return row;
}

static void parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--n") o.n = std::atoll(next("--n"));
        else if (a == "--chunk") o.chunk = std::atoll(next("--chunk"));
        else if (a == "--full") o.full = true;
        else if (a == "--block") o.block = std::atoi(next("--block"));
        else if (a == "--reps") o.reps = std::atoi(next("--reps"));
        else if (a == "--no-micro") o.micro = false;
        else if (a == "--micro-n") o.micro_n = std::atoll(next("--micro-n"));
        else if (a == "--micro-iters") o.micro_iters = std::atoi(next("--micro-iters"));
        else if (a == "--csv") o.csv = next("--csv");
        else {
            std::fprintf(stderr,
                         "usage: %s [--n N] [--chunk C | --full] [--block B] [--reps R]\n"
                         "          [--no-micro] [--micro-n N] [--micro-iters K] [--csv file]\n",
                         argv[0]);
            std::exit(a == "--help" ? 0 : 2);
        }
    }
    if (o.n <= 0 || o.block <= 0 || o.block > 1024 || o.reps <= 0) {
        std::fprintf(stderr, "bad arguments\n");
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    Options o;
    parse_args(argc, argv, o);

    CUDA_CHECK(cudaFree(0));  // инициализация контекста вне замеров
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int mem_clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double peak_gbps = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("GPU: %s, sm_%d%d, %.1f GiB (free %.1f GiB), peak mem BW ~%.0f GB/s\n", p.name,
                p.major, p.minor, total_b / 1073741824.0, free_b / 1073741824.0, peak_gbps);

    // Размер чанка: для double нужно 8 байт/элемент на устройстве.
    long long chunk = o.full ? o.n : std::min(o.n, o.chunk);
    long long max_by_mem = (long long)(0.85 * (double)free_b / sizeof(double));
    if (chunk > max_by_mem) {
        std::printf("note: chunk %lld does not fit GPU memory, reduced to %lld\n", chunk,
                    max_by_mem);
        chunk = max_by_mem;
    }
    chunk = std::min<long long>(chunk, 1LL << 31);  // индексы в ядрах 32-битные
    std::printf("N = %lld, chunk = %lld (%lld chunks), block = %d, reps = %d\n\n", o.n, chunk,
                (o.n + chunk - 1) / chunk, o.block, o.reps);

    // Эталоны: зависят только от i % 360.
    double ref_true[360], ref_fn_float[360];
    for (int r = 0; r < 360; ++r) {
        double a = (double)r * DEG2RAD;
        ref_true[r] = std::sin(a);
        ref_fn_float[r] = std::sin((double)(float)a);  // sin от реально поданного float-аргумента
    }

    void* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, (size_t)chunk * sizeof(double)));
    std::unique_ptr<char[]> h_raw(new char[(size_t)chunk * sizeof(double)]);
    float* h_f = reinterpret_cast<float*>(h_raw.get());
    double* h_d = reinterpret_cast<double*>(h_raw.get());

    auto grid = [](unsigned n, int block) { return (n + block - 1) / block; };
    std::vector<Row> rows;
    rows.push_back(run_variant<float>(
        "float", "sin (via double)",
        [&](void* d, unsigned n, unsigned base, int b) {
            fill_float<Fn::SinViaDouble><<<grid(n, b), b>>>((float*)d, n, base);
        },
        o, chunk, d_buf, h_f, ref_true, ref_fn_float));
    rows.push_back(run_variant<float>(
        "float", "sinf",
        [&](void* d, unsigned n, unsigned base, int b) {
            fill_float<Fn::Sinf><<<grid(n, b), b>>>((float*)d, n, base);
        },
        o, chunk, d_buf, h_f, ref_true, ref_fn_float));
    rows.push_back(run_variant<float>(
        "float", "__sinf",
        [&](void* d, unsigned n, unsigned base, int b) {
            fill_float<Fn::FastSinf><<<grid(n, b), b>>>((float*)d, n, base);
        },
        o, chunk, d_buf, h_f, ref_true, ref_fn_float));
    rows.push_back(run_variant<double>(
        "double", "sin",
        [&](void* d, unsigned n, unsigned base, int b) {
            fill_double<<<grid(n, b), b>>>((double*)d, n, base);
        },
        o, chunk, d_buf, h_d, ref_true, ref_true));

    std::printf("%-7s %-17s %-12s %-12s %-11s %-11s %-11s %-10s %-10s %s\n", "type", "function",
                "err_true", "err_fn", "maxerr_true", "maxerr_fn", "kernel,ms", "d2h,ms", "total,ms",
                "kernel GB/s");
    for (const Row& r : rows)
        std::printf("%-7s %-17s %-12.4e %-12.4e %-11.3e %-11.3e %-11.2f %-10.2f %-10.2f %.0f\n",
                    r.type.c_str(), r.fn.c_str(), r.err_true, r.err_fn, r.max_true, r.max_fn,
                    r.kernel_mean, r.d2h_mean, r.total_mean, r.gbps);
    std::printf("\nerr_true : |sin(i%%360*pi/180) [double] - arr[i]|, mean over N (as in the task)\n"
                "err_fn   : |sin((double)x_float) - arr[i]|, error of the function itself\n"
                "kernel,ms: sum over chunks, mean of %d reps; GB/s from the fastest rep\n\n",
                o.reps);

    // ---- Микро-бенчмарк: вычисления вместо памяти ----
    struct Micro {
        std::string name;
        double ms;
        double gsin;
    };
    std::vector<Micro> micro;
    if (o.micro) {
        unsigned mn = (unsigned)o.micro_n;
        void* d_m = nullptr;
        CUDA_CHECK(cudaMalloc(&d_m, (size_t)mn * sizeof(double)));
        auto time_it = [&](const char* name, const std::function<void()>& f) {
            f();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> t;
            EventTimer et;
            for (int r = 0; r < o.reps; ++r) {
                et.start();
                f();
                t.push_back(et.stop());
                launch_check();
            }
            double ms = mean_of(t);
            micro.push_back({name, ms, (double)mn * o.micro_iters / (ms * 1e-3) / 1e9});
        };
        int b = o.block;
        int it = o.micro_iters;
        time_it("float sin (via double)", [&] {
            bench_float<Fn::SinViaDouble><<<grid(mn, b), b>>>((float*)d_m, mn, it);
        });
        time_it("float sinf", [&] { bench_float<Fn::Sinf><<<grid(mn, b), b>>>((float*)d_m, mn, it); });
        time_it("float __sinf",
                [&] { bench_float<Fn::FastSinf><<<grid(mn, b), b>>>((float*)d_m, mn, it); });
        time_it("double sin", [&] { bench_double<<<grid(mn, b), b>>>((double*)d_m, mn, it); });
        CUDA_CHECK(cudaFree(d_m));

        std::printf("Compute-bound micro-benchmark: %u elements x %d dependent calls each\n", mn,
                    it);
        std::printf("%-24s %-12s %s\n", "variant", "time,ms", "Gcalls/s");
        for (const Micro& m : micro)
            std::printf("%-24s %-12.2f %.1f\n", m.name.c_str(), m.ms, m.gsin);
    }

    if (!o.csv.empty()) {
        FILE* f = std::fopen(o.csv.c_str(), "w");
        if (!f) {
            std::perror("csv");
        } else {
            std::fprintf(f,
                         "gpu,n,chunk,block,type,function,err_true,err_fn,maxerr_true,maxerr_fn,"
                         "kernel_ms_mean,kernel_ms_min,d2h_ms_mean,total_ms_mean,kernel_gbps\n");
            for (const Row& r : rows)
                std::fprintf(f, "\"%s\",%lld,%lld,%d,%s,%s,%.6e,%.6e,%.6e,%.6e,%.4f,%.4f,%.4f,%.4f,%.1f\n",
                             p.name, o.n, chunk, o.block, r.type.c_str(), r.fn.c_str(), r.err_true,
                             r.err_fn, r.max_true, r.max_fn, r.kernel_mean, r.kernel_min,
                             r.d2h_mean, r.total_mean, r.gbps);
            std::fclose(f);
            std::printf("\nCSV written to %s\n", o.csv.c_str());
        }
    }

    CUDA_CHECK(cudaFree(d_buf));
    return 0;
}
