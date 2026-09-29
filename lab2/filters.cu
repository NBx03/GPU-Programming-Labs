// ЛР2. Фильтры изображений на CUDA: размытие по Гауссу, выделение границ (Собель), медианный фильтр.
//
// Конвейер: PNG (libpng) -> массив RGB на хосте -> cudaMalloc/cudaMemcpy H2D -> ядро ->
//           cudaMemcpy D2H -> массив -> PNG.
// Для каждого фильтра есть два варианта ядра: "global" (обращения в глобальную память) и
// "shared" (тайл с гало в разделяемой памяти). CPU-эталон использует те же функции пикселя
// (filter_common.h) и собирается без FMA, GPU -- с --fmad=false, поэтому результаты совпадают.
//
// Использование:
//   filters <in.png> <out.png> <blur|sobel|median> [--radius R] [--variant global|shared|both]
//           [--block B] [--reps N] [--cpu-threads T]
//   filters --bench [in.png] [--sizes 512,2048,4096] [--radii 1,2,4,7] [--reps N] [--block B]
//           [--csv file]
//   filters --gen S out.png            (сгенерировать тестовое изображение SxS)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/cuda_check.h"
#include "cpu_filters.h"
#include "filter_common.h"
#include "image_io.h"

// ------------------------------------------------------------------------------------------
// Ядра
// ------------------------------------------------------------------------------------------
__constant__ float c_w[2 * MAX_RADIUS + 1];

// Загрузка тайла RGB (с гало) из глобальной памяти в разделяемую. Координаты -- clamp.
__device__ __forceinline__ void load_tile_rgb(uint8_t* tile, const uint8_t* in, int w, int h,
                                              int x0, int y0, int tw, int th) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nt = blockDim.x * blockDim.y;
    for (int p = tid; p < tw * th; p += nt) {
        int ty = p / tw, tx = p - ty * tw;
        const uint8_t* s = in + ((size_t)clampi(y0 + ty, 0, h - 1) * w + clampi(x0 + tx, 0, w - 1)) * 3;
        tile[p * 3 + 0] = s[0];
        tile[p * 3 + 1] = s[1];
        tile[p * 3 + 2] = s[2];
    }
}

// ---- Размытие по Гауссу (2D-свёртка (2r+1)x(2r+1), веса в constant memory) ----
__global__ void blur_global(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int w, int h,
                            int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        float acc = 0.f;
        for (int dy = -r; dy <= r; ++dy) {
            int yy = clampi(y + dy, 0, h - 1);
            float wy = c_w[dy + r];
            for (int dx = -r; dx <= r; ++dx) {
                int xx = clampi(x + dx, 0, w - 1);
                acc += (wy * c_w[dx + r]) * (float)in[((size_t)yy * w + xx) * 3 + c];
            }
        }
        out[((size_t)y * w + x) * 3 + c] = to_u8(acc);
    }
}

__global__ void blur_shared(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int w, int h,
                            int r) {
    extern __shared__ uint8_t smem[];
    int tw = blockDim.x + 2 * r, th = blockDim.y + 2 * r;
    load_tile_rgb(smem, in, w, h, blockIdx.x * blockDim.x - r, blockIdx.y * blockDim.y - r, tw, th);
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        float acc = 0.f;
        for (int dy = -r; dy <= r; ++dy) {
            int ty = threadIdx.y + r + dy;
            float wy = c_w[dy + r];
            for (int dx = -r; dx <= r; ++dx) {
                int tx = threadIdx.x + r + dx;
                acc += (wy * c_w[dx + r]) * (float)smem[(ty * tw + tx) * 3 + c];
            }
        }
        out[((size_t)y * w + x) * 3 + c] = to_u8(acc);
    }
}

// ---- Собель: яркость -> градиенты -> модуль; результат -- серое изображение в RGB ----
__global__ void sobel_global(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int l[9];
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const uint8_t* p = in + ((size_t)clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)) * 3;
            l[(dy + 1) * 3 + (dx + 1)] = luma(p[0], p[1], p[2]);
        }
    uint8_t v = sobel_from_luma(l);
    uint8_t* o = out + ((size_t)y * w + x) * 3;
    o[0] = o[1] = o[2] = v;
}

__global__ void sobel_shared(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int w, int h) {
    extern __shared__ uint8_t smem[];  // тайл яркостей (bx+2) x (by+2)
    int tw = blockDim.x + 2, th = blockDim.y + 2;
    int x0 = blockIdx.x * blockDim.x - 1, y0 = blockIdx.y * blockDim.y - 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x, nt = blockDim.x * blockDim.y;
    for (int p = tid; p < tw * th; p += nt) {
        int ty = p / tw, tx = p - ty * tw;
        const uint8_t* s = in + ((size_t)clampi(y0 + ty, 0, h - 1) * w + clampi(x0 + tx, 0, w - 1)) * 3;
        smem[p] = (uint8_t)luma(s[0], s[1], s[2]);
    }
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int l[9];
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            l[(dy + 1) * 3 + (dx + 1)] = smem[(threadIdx.y + 1 + dy) * tw + threadIdx.x + 1 + dx];
    uint8_t v = sobel_from_luma(l);
    uint8_t* o = out + ((size_t)y * w + x) * 3;
    o[0] = o[1] = o[2] = v;
}

// ---- Медианный фильтр 3x3 по каждому каналу ----
__global__ void median_global(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        uint8_t v[9];
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                v[(dy + 1) * 3 + (dx + 1)] =
                    in[((size_t)clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)) * 3 + c];
        out[((size_t)y * w + x) * 3 + c] = median9(v);
    }
}

__global__ void median_shared(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int w, int h) {
    extern __shared__ uint8_t smem[];
    int tw = blockDim.x + 2, th = blockDim.y + 2;
    load_tile_rgb(smem, in, w, h, blockIdx.x * blockDim.x - 1, blockIdx.y * blockDim.y - 1, tw, th);
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        uint8_t v[9];
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                v[(dy + 1) * 3 + (dx + 1)] = smem[((threadIdx.y + 1 + dy) * tw + threadIdx.x + 1 + dx) * 3 + c];
        out[((size_t)y * w + x) * 3 + c] = median9(v);
    }
}

// ------------------------------------------------------------------------------------------
// Хост
// ------------------------------------------------------------------------------------------
enum class Filter { Blur, Sobel, Median };
enum class Variant { Global, Shared };

static const char* variant_name(Variant v) { return v == Variant::Global ? "global" : "shared"; }

static std::string filter_label(Filter f, int r) {
    switch (f) {
        case Filter::Blur: return "blur r=" + std::to_string(r);
        case Filter::Sobel: return "sobel";
        default: return "median";
    }
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct GpuTimes {
    double alloc = 0, h2d = 0, kernel = 0, d2h = 0;  // мс
    double pipeline() const { return alloc + h2d + kernel + d2h; }
};

static int g_smem_limit = 48 * 1024;

// Один полный проход GPU-конвейера (без файлового ввода-вывода) с разбивкой по этапам.
static void gpu_filter(Filter f, Variant var, int radius, int block, const uint8_t* h_in,
                       uint8_t* h_out, int w, int h, GpuTimes& t) {
    const size_t bytes = (size_t)w * h * 3;
    uint8_t *d_in = nullptr, *d_out = nullptr;

    auto t0 = Clock::now();
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    t.alloc = ms_since(t0);

    EventTimer et;
    et.start();
    if (f == Filter::Blur) {
        float wt[2 * MAX_RADIUS + 1];
        make_gauss_weights(radius, wt);
        CUDA_CHECK(cudaMemcpyToSymbol(c_w, wt, (2 * radius + 1) * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    t.h2d = et.stop();

    dim3 blk(block, block), grid((w + block - 1) / block, (h + block - 1) / block);
    size_t smem = 0;
    if (var == Variant::Shared) {
        size_t tw = block + (f == Filter::Blur ? 2 * radius : 2);
        smem = tw * tw * (f == Filter::Sobel ? 1 : 3);
        if ((int)smem > g_smem_limit) {
            std::fprintf(stderr, "shared tile %zu B exceeds limit %d B; use smaller --block/--radius\n",
                         smem, g_smem_limit);
            std::exit(2);
        }
    }
    et.start();
    if (f == Filter::Blur) {
        if (var == Variant::Global) blur_global<<<grid, blk>>>(d_in, d_out, w, h, radius);
        else blur_shared<<<grid, blk, smem>>>(d_in, d_out, w, h, radius);
    } else if (f == Filter::Sobel) {
        if (var == Variant::Global) sobel_global<<<grid, blk>>>(d_in, d_out, w, h);
        else sobel_shared<<<grid, blk, smem>>>(d_in, d_out, w, h);
    } else {
        if (var == Variant::Global) median_global<<<grid, blk>>>(d_in, d_out, w, h);
        else median_shared<<<grid, blk, smem>>>(d_in, d_out, w, h);
    }
    t.kernel = et.stop();
    CUDA_CHECK(cudaGetLastError());

    et.start();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    t.d2h = et.stop();

    t0 = Clock::now();
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    t.alloc += ms_since(t0);  // cudaMalloc + cudaFree
}

// Медианы этапов по reps запускам (+1 прогревочный).
static GpuTimes gpu_median(Filter f, Variant var, int radius, int block, const Image& in,
                           std::vector<uint8_t>& out, int reps) {
    out.assign(in.rgb.size(), 0);
    GpuTimes t;
    gpu_filter(f, var, radius, block, in.rgb.data(), out.data(), in.w, in.h, t);  // прогрев
    std::vector<double> a, b, c, d;
    for (int i = 0; i < reps; ++i) {
        gpu_filter(f, var, radius, block, in.rgb.data(), out.data(), in.w, in.h, t);
        a.push_back(t.alloc);
        b.push_back(t.h2d);
        c.push_back(t.kernel);
        d.push_back(t.d2h);
    }
    return {median_of(a), median_of(b), median_of(c), median_of(d)};
}

static double cpu_median_time(Filter f, int radius, int threads, const Image& in,
                              std::vector<uint8_t>& out, int reps) {
    out.assign(in.rgb.size(), 0);
    std::vector<double> ts;
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        if (f == Filter::Blur) cpu_blur(in.rgb.data(), out.data(), in.w, in.h, radius, threads);
        else if (f == Filter::Sobel) cpu_sobel(in.rgb.data(), out.data(), in.w, in.h, threads);
        else cpu_median(in.rgb.data(), out.data(), in.w, in.h, threads);
        ts.push_back(ms_since(t0));
    }
    return median_of(ts);
}

static void compare(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, long long& ndiff,
                    int& maxdiff) {
    ndiff = 0;
    maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int d = std::abs((int)a[i] - (int)b[i]);
        if (d) ++ndiff;
        if (d > maxdiff) maxdiff = d;
    }
}

static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> r;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t c = s.find(',', pos);
        if (c == std::string::npos) c = s.size();
        r.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
        pos = c + 1;
    }
    return r;
}

static void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s <in.png> <out.png> <blur|sobel|median> [--radius R] [--variant global|shared|both]\n"
                 "        [--block B] [--reps N] [--cpu-threads T]\n"
                 "  %s --bench [in.png] [--sizes 512,2048,4096] [--radii 1,2,4,7] [--reps N] [--block B]\n"
                 "        [--csv file]\n"
                 "  %s --gen S out.png\n",
                 a0, a0, a0);
}

static void print_device() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int lim = 0;
    cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    if (lim > 0) g_smem_limit = lim;
    std::printf("GPU: %s (sm_%d%d), %d SM, %.1f GiB | CPU threads: %d\n", p.name, p.major, p.minor,
                p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, cpu_max_threads());
}

// ------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    std::string a1 = argv[1];
    std::string err;

    if (a1 == "--gen") {
        if (argc < 4) {
            usage(argv[0]);
            return 2;
        }
        int s = std::atoi(argv[2]);
        if (s <= 0 || !write_png(argv[3], make_test_image(s, s), err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d)\n", argv[3], s, s);
        return 0;
    }

    // ---- разбор опций ----
    bool bench = (a1 == "--bench");
    std::string in_path, out_path, csv_path, sizes_s = "512,2048,4096", radii_s = "1,2,4,7",
                                              variant_s = "both";
    Filter filter = Filter::Blur;
    int radius = 2, block = 16, reps = 5, cpu_threads = 0;
    std::vector<std::string> pos;
    for (int i = bench ? 2 : 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--radius") radius = std::atoi(next());
        else if (a == "--variant") variant_s = next();
        else if (a == "--block") block = std::atoi(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next());
        else if (a == "--sizes") sizes_s = next();
        else if (a == "--radii") radii_s = next();
        else if (a == "--csv") csv_path = next();
        else if (a.rfind("--", 0) == 0) {
            usage(argv[0]);
            return 2;
        } else pos.push_back(a);
    }
    if (block < 1 || block > 32 || reps < 1 || radius < 1 || radius > MAX_RADIUS) {
        std::fprintf(stderr, "bad arguments (block 1..32, reps>=1, radius 1..%d)\n", MAX_RADIUS);
        return 2;
    }
    if (cpu_threads <= 0) cpu_threads = cpu_max_threads();

    CUDA_CHECK(cudaFree(0));  // инициализация контекста вне замеров
    print_device();

    std::vector<Variant> variants;
    if (variant_s == "global" || variant_s == "both") variants.push_back(Variant::Global);
    if (variant_s == "shared" || variant_s == "both") variants.push_back(Variant::Shared);
    if (variants.empty()) {
        usage(argv[0]);
        return 2;
    }

    // =========================== режим бенчмарка ===========================
    if (bench) {
        Image base;
        bool have_in = !pos.empty();
        if (have_in && !read_png(pos[0], base, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::vector<int> sizes = parse_list(sizes_s), radii = parse_list(radii_s);
        FILE* csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
        const char* hdr =
            "size,filter,variant,decode_ms,encode_ms,cpu1_ms,cpuN_ms,alloc_ms,h2d_ms,kernel_ms,"
            "d2h_ms,gpu_pipeline_ms,speedup_kernel,speedup_pipeline,speedup_e2e,speedup_vs_cpuN,"
            "diff_pixels,max_diff";
        if (csv) std::fprintf(csv, "%s\n", hdr);
        std::printf("\n%-6s %-9s %-7s | %8s %8s | %9s %9s | %7s %7s %8s %7s %9s | %8s %8s %8s %8s | %s\n",
                    "size", "filter", "variant", "decode", "encode", "cpu 1T", "cpu MT", "alloc",
                    "h2d", "kernel", "d2h", "gpu total", "x kernel", "x pipe", "x e2e", "x cpuMT",
                    "diff");
        std::printf("(times in ms; x = speedup vs CPU 1 thread; e2e = decode + compute + encode)\n");

        for (int S : sizes) {
            Image img = have_in ? resize_nearest(base, S, S) : make_test_image(S, S);
            const char* tmp = "bench_tmp.png";
            auto t0 = Clock::now();
            if (!write_png(tmp, img, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            double enc = ms_since(t0);
            Image back;
            t0 = Clock::now();
            if (!read_png(tmp, back, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            double dec = ms_since(t0);
            std::remove(tmp);

            struct Case {
                Filter f;
                int r;
            };
            std::vector<Case> cases;
            for (int r : radii) cases.push_back({Filter::Blur, r});
            cases.push_back({Filter::Sobel, 0});
            cases.push_back({Filter::Median, 0});

            const bool big = (long long)S * S >= 4000000LL;
            for (const Case& cs : cases) {
                if (cs.f == Filter::Blur && (cs.r < 1 || cs.r > MAX_RADIUS)) continue;
                std::vector<uint8_t> ref1, refN, gout;
                int cpu_reps = big ? 1 : std::min(reps, 3);
                double cpu1 = cpu_median_time(cs.f, cs.r, 1, back, ref1, cpu_reps);
                double cpuN = cpu_median_time(cs.f, cs.r, cpu_threads, back, refN, cpu_reps);
                for (Variant v : variants) {
                    GpuTimes g = gpu_median(cs.f, v, cs.r, block, back, gout, reps);
                    long long nd;
                    int md;
                    compare(ref1, gout, nd, md);
                    double pipe = g.pipeline();
                    double s_k = cpu1 / g.kernel, s_p = cpu1 / pipe;
                    double s_e = (dec + cpu1 + enc) / (dec + pipe + enc), s_n = cpuN / pipe;
                    std::printf("%-6d %-9s %-7s | %8.1f %8.1f | %9.2f %9.2f | %7.2f %7.2f %8.3f %7.2f %9.2f | %8.1f %8.1f %8.2f %8.2f | %lld/%d\n",
                                S, filter_label(cs.f, cs.r).c_str(), variant_name(v), dec, enc, cpu1,
                                cpuN, g.alloc, g.h2d, g.kernel, g.d2h, pipe, s_k, s_p, s_e, s_n, nd, md);
                    std::fflush(stdout);
                    if (csv)
                        std::fprintf(csv,
                                     "%d,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.2f,%.2f,%.3f,%.3f,%lld,%d\n",
                                     S, filter_label(cs.f, cs.r).c_str(), variant_name(v), dec, enc,
                                     cpu1, cpuN, g.alloc, g.h2d, g.kernel, g.d2h, pipe, s_k, s_p, s_e,
                                     s_n, nd, md);
                }
            }
        }
        if (csv) {
            std::fclose(csv);
            std::printf("\nCSV written to %s\n", csv_path.c_str());
        }
        return 0;
    }

    // =========================== одиночный запуск ===========================
    if (pos.size() < 3) {
        usage(argv[0]);
        return 2;
    }
    in_path = pos[0];
    out_path = pos[1];
    if (pos[2] == "blur") filter = Filter::Blur;
    else if (pos[2] == "sobel") filter = Filter::Sobel;
    else if (pos[2] == "median") filter = Filter::Median;
    else {
        usage(argv[0]);
        return 2;
    }

    Image img;
    auto t0 = Clock::now();
    if (!read_png(in_path, img, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    double dec = ms_since(t0);
    std::printf("input %s: %dx%d, filter %s\n\n", in_path.c_str(), img.w, img.h,
                filter_label(filter, radius).c_str());

    std::vector<uint8_t> ref1, refN, gout, first_out;
    int cpu_reps = std::min(reps, 3);
    double cpu1 = cpu_median_time(filter, radius, 1, img, ref1, cpu_reps);
    double cpuN = cpu_median_time(filter, radius, cpu_threads, img, refN, cpu_reps);
    std::printf("PNG decode: %.1f ms | CPU 1 thread: %.2f ms | CPU %d threads: %.2f ms\n\n", dec, cpu1,
                cpu_threads, cpuN);
    std::printf("%-8s | %8s %8s %8s %8s | %9s | %8s %8s | %s\n", "variant", "alloc", "h2d", "kernel",
                "d2h", "gpu total", "x kernel", "x pipe", "GPU vs CPU: differing bytes / max diff");
    for (Variant v : variants) {
        GpuTimes g = gpu_median(filter, v, radius, block, img, gout, reps);
        long long nd;
        int md;
        compare(ref1, gout, nd, md);
        std::printf("%-8s | %8.2f %8.2f %8.3f %8.2f | %9.2f | %8.1f %8.1f | %lld / %d\n",
                    variant_name(v), g.alloc, g.h2d, g.kernel, g.d2h, g.pipeline(), cpu1 / g.kernel,
                    cpu1 / g.pipeline(), nd, md);
        if (first_out.empty()) first_out = gout;
    }

    Image res;
    res.w = img.w;
    res.h = img.h;
    res.rgb = first_out;
    t0 = Clock::now();
    if (!write_png(out_path, res, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::printf("\nPNG encode+write: %.1f ms -> %s\n", ms_since(t0), out_path.c_str());
    return 0;
}
